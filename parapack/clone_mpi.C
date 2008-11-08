/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2008 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

#include "clone_mpi.h"
#include "clone_info_mpi.h"
#include "parallel_factory.h"
#include "util.h"
#include <boost/filesystem/operations.hpp>

namespace alps {

clone_mpi::clone_mpi(boost::mpi::communicator const& world_d,
  boost::mpi::communicator const& world_u, boost::mpi::communicator const& work,
  clone_create_msg_t const& msg)
  : world_d_(world_d), world_u_(world_u), work_(work) {

  task_id_ = msg.task_id;
  clone_id_ = msg.clone_id;
  group_id_ = msg.group_id;
  params_ = msg.params;
  basedir_ = boost::filesystem::path(msg.basedir);
  bool is_new = msg.is_new;

  params_["DIR_NAME"] = msg.basedir;
  params_["BASE_NAME"] = msg.base;

  info_ = clone_info_mpi(work_, clone_id_, params_, msg.base);
  params_["WORKER_SEED"] = info_.worker_seed();
  params_["DISORDER_SEED"] = info_.disorder_seed();

  if (work.size() > 1)
    worker_ = alps::parapack::parallel_worker_factory::make_worker(work, params_);
  else
    worker_ = alps::parapack::worker_factory::make_worker(params_);
  if (is_new) worker_->init_observables(params_, measurements_);
  if (!is_new) {
    IXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
    this->load(dp);
    send_info(mcmp_tag::clone_info);
  }

  if (work_.rank() == 0 && is_new && worker_->is_thermalized()) {
    // no thermalization steps
    BOOST_FOREACH(alps::ObservableSet& m, measurements_) m.reset(true);
  }

  if (is_new || info_.progress() < 1)
    info_.start((worker_->is_thermalized()) ? "running" : "equilibrating");
  if (work_.rank() == 0 && is_new && worker_->progress() >= 1) {
    info_.set_progress(worker_->progress());
    info_.stop();
    send_info(mcmp_tag::clone_info);
  }
}

clone_mpi::~clone_mpi() {}

void clone_mpi::run() {
  bool thermalized = worker_->is_thermalized();
  double progress = worker_->progress();

  int tag = (progress < 1) ? mcmp_tag::do_step : mcmp_tag::do_nothing;
  if (work_.rank() == 0 && world_d_.iprobe(0, boost::mpi::any_tag)) {
    boost::mpi::status stat = world_d_.recv(0, boost::mpi::any_tag);
    tag = stat.tag();
    if (tag == mcmp_tag::clone_suspend && info_.progress() >= 1)
      tag = mcmp_tag::clone_halt;
  }
  broadcast(work_, tag, 0);

  switch (tag) {
  case mcmp_tag::do_step :
    worker_->run(measurements_);
    break;
  case mcmp_tag::clone_info :
    send_info(mcmp_tag::clone_info);
    break;
  case mcmp_tag::clone_checkpoint :
    checkpoint();
    send_info(mcmp_tag::clone_checkpoint);
    break;
  case mcmp_tag::clone_suspend :
    this->suspend();
    send_info(mcmp_tag::clone_suspend);
    break;
  case mcmp_tag::clone_halt :
    this->halt();
    send_halted();
    break;
  case mcmp_tag::do_nothing :
    break;
  default:
    if (work_.rank() == 0)
      std::cerr << "Warning: ignoring a message with an unknown tag " << tag << std::endl;
  }

  if (worker_) {
    if (!thermalized && worker_->is_thermalized()) {
      BOOST_FOREACH(alps::ObservableSet& m, measurements_) m.reset(true);
    }
    if (work_.rank() == 0) {
      info_.set_progress(worker_->progress());
      if (!thermalized && worker_->is_thermalized()) {
        info_.stop();
        info_.start("running");
      }
      if (progress < 1 && info_.progress() >= 1) {
        info_.stop();
        send_info(mcmp_tag::clone_info);
      }
    }
  }
}

void clone_mpi::checkpoint() {
  if (work_.rank() == 0 && info_.progress() < 1) info_.stop();
  OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
  this->save(dp);
}

void clone_mpi::suspend() {
  if (work_.rank() == 0) info_.stop();
  OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
  this->save(dp);
  work_.barrier();
  worker_.reset();
}

void clone_mpi::halt() {
  if (work_.rank() == 0 && info_.progress() < 1)
    boost::throw_exception(std::logic_error("clone is not finished"));
  OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
  this->save(dp);
  work_.barrier();
  worker_.reset();
}

bool clone_mpi::halted() const { return !worker_; }

double clone_mpi::progress() const { return info_.progress(); }

clone_info const& clone_mpi::info() const { return info_; }

void clone_mpi::load(IDump& dp) {
  int32_t tag;
  dp >> tag;
  if (work_.rank() == 0) {
    if (tag != parapack_dump::run_master)
      boost::throw_exception(std::runtime_error("dump does not contain a master clone"));
  } else {
    if (tag != parapack_dump::run_slave)
      boost::throw_exception(std::runtime_error("dump does not contain a slave clone"));
  }

  int32_t version, np, rank;
  dp >> version >> np >> rank;
  dp.set_version(version);

  if (version < parapack_dump::initial_version || version > parapack_dump::current_version)
    boost::throw_exception(std::runtime_error("The clone on dump is version " +
      boost::lexical_cast<std::string>(version) +
      " but this program can read only versions between " +
      boost::lexical_cast<std::string>(parapack_dump::initial_version) + " and " +
      boost::lexical_cast<std::string>(parapack_dump::current_version)));
  if (np != work_.size())
    boost::throw_exception(std::runtime_error("inconsistent number of processes"));
  if (rank != work_.rank())
    boost::throw_exception(std::runtime_error("inconsistent rank of process"));

  if (version >= 304) {
    dp >> params_ >> info_ >> measurements_;
  } else {
    measurements_.resize(1);
    dp >> params_ >> info_ >> measurements_[0];
  }

  bool full_dump(dp);
  if (full_dump) worker_->load_worker(dp);
}

void clone_mpi::save(ODump& dp) const{
  if (work_.rank() == 0)
    dp << static_cast<int32_t>(parapack_dump::run_master);
  else
    dp << static_cast<int32_t>(parapack_dump::run_slave);

  dp << static_cast<int32_t>(parapack_dump::current_version)
     << static_cast<int32_t>(work_.size())
     << static_cast<int32_t>(work_.rank());

  dp << params_ << info_ << measurements_;

  bool full_dump = false;
  if (work_.rank() == 0)
    full_dump =
      (info_.progress() < 1) || params_.value_or_default("SCHEDULER_KEEP_FULL_DUMP", false);
  broadcast(work_, full_dump, 0);
  dp << full_dump;
  if (full_dump) worker_->save_worker(dp);
}

void clone_mpi::output() const{
  std::cout << params_;
  BOOST_FOREACH(ObservableSet const& m, measurements_) std::cout << m;
}

void clone_mpi::send_info(mcmp_tag_t tag) {
  if (work_.rank() == 0)
    world_u_.send(0, tag, clone_info_msg_t(task_id_, clone_id_, group_id_, info_));
}

void clone_mpi::send_halted() {
  if (work_.rank() == 0)
    world_u_.send(0, mcmp_tag::clone_halt, clone_halt_msg_t(task_id_, clone_id_, group_id_));
}

} // end namespace alps
