/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
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

#include "clone.h"
#include <boost/filesystem/operations.hpp>

namespace alps {

void save_observable(alps::hdf5::oarchive& ar, std::string const& prefix,
                     std::vector<ObservableSet> const& obs) {
  if (obs.size() == 1)
    ar << make_pvp(prefix, obs[0]);
  else
    for (int m = 0; m < obs.size(); ++m)
      ar << make_pvp(prefix + "/sections/" + boost::lexical_cast<std::string>(m), obs[m]);
}

void save_observable(alps::hdf5::oarchive& ar, std::vector<ObservableSet> const& obs) {
  save_observable(ar, "simulation/results", obs);
}

void save_observable(alps::hdf5::oarchive& ar, cid_t cid, std::vector<ObservableSet> const& obs) {
  save_observable(ar, "simulation/realizations/" + boost::lexical_cast<std::string>(0) +
                  "/clones/" + boost::lexical_cast<std::string>(cid) + "/results", obs);
}

void save_observable(alps::hdf5::oarchive& ar, cid_t cid, int rank,
                     std::vector<ObservableSet> const& obs) {
  save_observable(ar, "simulation/realizations/" + boost::lexical_cast<std::string>(0) +
                  "/clones/" + boost::lexical_cast<std::string>(cid) +
                  "/workers/" + boost::lexical_cast<std::string>(rank) + "/results", obs);
}

bool load_observable(alps::hdf5::iarchive& ar, std::string const& prefix,
                     std::vector<ObservableSet>& obs) {
  obs.clear();
  if (ar.is_group(prefix)) {
    if (!ar.is_group(prefix + "/sections/0")) {
      obs.resize(1);
      ar >> make_pvp(prefix, obs[0]);
    } else {
      for (int m = 0; ; ++m) {
        std::string p = prefix + "/sections/" + boost::lexical_cast<std::string>(m);
        if (ar.is_group(p)) {
          obs.push_back(ObservableSet());
          ar >> make_pvp(p, obs[m]);
        } else {
          break;
        }
      }
    }
    return true;
  } else {
    return false;
  }
}

bool load_observable(alps::hdf5::iarchive& ar, std::vector<ObservableSet>& obs) {
  return load_observable(ar, "simulation/results", obs);
}

bool load_observable(alps::hdf5::iarchive& ar, cid_t cid, std::vector<ObservableSet>& obs) {
  return load_observable(ar, "simulation/realizations/" + boost::lexical_cast<std::string>(0) +
                         "/clones/" + boost::lexical_cast<std::string>(cid) + "/results", obs);
}

bool load_observable(alps::hdf5::iarchive& ar, cid_t cid, int rank,
                       std::vector<ObservableSet>& obs) {
  return load_observable(ar, "simulation/realizations/" + boost::lexical_cast<std::string>(0) +
                         "/clones/" + boost::lexical_cast<std::string>(cid) +
                         "/workers/" + boost::lexical_cast<std::string>(rank) + "/results", obs);
}


//
// clone
//

clone::clone(tid_t tid, cid_t cid, Parameters const& params, boost::filesystem::path const& basedir,
  std::string const& base, clone_timer::duration_t const& check_interval, bool is_new)
  : task_id_(tid), clone_id_(cid), params_(params), basedir_(basedir), timer_(check_interval) {
  params_["DIR_NAME"] = basedir_.native_file_string();
  params_["BASE_NAME"] = base;

  info_ = clone_info(clone_id_, params_, base);
  params_["WORKER_SEED"] = info_.worker_seed();
  params_["DISORDER_SEED"] = info_.disorder_seed();

  worker_ = parapack::worker_factory::make_worker(params_);
  if (is_new) worker_->init_observables(params_, measurements_);
  if (!is_new) {
    this->load();
  }

  if (is_new && worker_->is_thermalized()) { // no thermalization steps
    BOOST_FOREACH(alps::ObservableSet& m, measurements_) { m.reset(true); }
  }

  if (is_new || worker_->progress() < 1)
    info_.start((worker_->is_thermalized()) ? "running" : "equilibrating");
  if (is_new && worker_->progress() >= 1) {
    info_.set_progress(worker_->progress());
    info_.stop();
    do_halt();
  }

  if (!is_new) timer_.reset(worker_->progress());
  loops_ = 1;
}

clone::~clone() {}

void clone::run() {
  for (clone_timer::loops_t i = 0; i < loops_; ++i) {
    bool thermalized = worker_->is_thermalized();
    double progress = worker_->progress();
    worker_->run(measurements_);
    if (!thermalized && worker_->is_thermalized()) {
      BOOST_FOREACH(alps::ObservableSet& m, measurements_) m.reset(true);
      info_.stop();
      info_.start("running");
    }
    if (progress < 1 && worker_->progress() >= 1) {
      info_.set_progress(worker_->progress());
      info_.stop();
      do_halt();
      return;
    }
  }
  info_.set_progress(worker_->progress());
  loops_ = timer_.next_loops(loops_);
}

bool clone::halted() const { return !worker_; }

clone_info const& clone::info() const { return info_; }

void clone::load() {
  {
    IXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
    worker_->load_worker(dp);
  }
  #pragma omp critical (hdf5io)
  {
    hdf5::iarchive h5(complete(boost::filesystem::path(info_.dumpfile_h5()), basedir_));
    h5 >> make_pvp("/", this);
  }
}

void clone::save() const{
  {
    OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
    worker_->save_worker(dp);
  }
  #pragma omp critical (hdf5io)
  {
    hdf5::oarchive h5(complete(boost::filesystem::path(info_.dumpfile_h5()), basedir_));
    h5 << make_pvp("/", this);
  }
}

void clone::serialize(hdf5::iarchive& ar) {
  ar >> make_pvp("parameters", params_)
     >> make_pvp("log/alps", info_);
  load_observable(ar, clone_id_, measurements_);
}

void clone::serialize(hdf5::oarchive& ar) const {
  ar << make_pvp("parameters", params_)
     << make_pvp("log/alps", info_);
  save_observable(ar, clone_id_, measurements_);
}

void clone::checkpoint() {
  if (info_.progress() < 1) info_.stop();
  this->save();
}

void clone::suspend() {
  info_.stop();
  this->save();
  worker_.reset();
}

void clone::do_halt() {
  if (info_.progress() < 1)
    boost::throw_exception(std::logic_error("clone is not finished"));
  this->save();
  worker_.reset();
}

void clone::output() const{
  std::cout << params_;
  BOOST_FOREACH(ObservableSet const& m, measurements_) std::cout << m;
}

#ifdef ALPS_HAVE_MPI

//
// clone_mpi
//

clone_mpi::clone_mpi(boost::mpi::communicator const& ctrl,
  boost::mpi::communicator const& work, clone_create_msg_t const& msg)
  : ctrl_(ctrl), work_(work), task_id_(msg.task_id), clone_id_(msg.clone_id),
    group_id_(msg.group_id), params_(msg.params), basedir_(boost::filesystem::path(msg.basedir)),
    timer_(msg.check_interval) {
  bool is_new = msg.is_new;

  params_["DIR_NAME"] = msg.basedir;
  params_["BASE_NAME"] = msg.base;

  info_ = clone_info_mpi(work_, clone_id_, params_, msg.base);
  params_["WORKER_SEED"] = info_.worker_seed();
  params_["DISORDER_SEED"] = info_.disorder_seed();

  if (work_.size() > 1)
    worker_ = alps::parapack::parallel_worker_factory::make_worker(work_, params_);
  else
    worker_ = alps::parapack::worker_factory::make_worker(params_);
  if (is_new) worker_->init_observables(params_, measurements_);
  if (!is_new) {
    this->load();
  }

  if (is_new && worker_->is_thermalized()) { // no thermalization steps
    BOOST_FOREACH(alps::ObservableSet& m, measurements_) { m.reset(true); }
  }

  if (is_new || info_.progress() < 1)
    info_.start((worker_->is_thermalized()) ? "running" : "equilibrating");
  if (work_.rank() == 0 && is_new && worker_->progress() >= 1) {
    info_.set_progress(worker_->progress());
    info_.stop();
    send_info(mcmp_tag::clone_info);
  }

  if (work_.rank() == 0 && !is_new) timer_.reset(worker_->progress());
  loops_ = 1;
}

clone_mpi::~clone_mpi() {}

void clone_mpi::run() {
  int tag = (info_.progress() < 1) ? mcmp_tag::do_step : mcmp_tag::do_nothing;
  if (work_.rank() == 0 && ctrl_.iprobe(0, boost::mpi::any_tag)) {
    boost::mpi::status stat = ctrl_.recv(0, boost::mpi::any_tag);
    tag = stat.tag();
    if (tag == mcmp_tag::clone_suspend && info_.progress() >= 1)
      tag = mcmp_tag::clone_halt;
  }
  broadcast(work_, tag, 0);

  switch (tag) {
  case mcmp_tag::do_step :
    for (clone_timer::loops_t i = 0; i < loops_; ++i) {
      bool thermalized = worker_->is_thermalized();
      double progress = worker_->progress();
      worker_->run(measurements_);
      if (!thermalized && worker_->is_thermalized()) {
        BOOST_FOREACH(alps::ObservableSet& m, measurements_) m.reset(true);
        if (work_.rank() == 0) {
          info_.stop();
          info_.start("running");
        }
      }
      if (progress < 1 && worker_->progress() >= 1) {
        if (work_.rank() == 0) {
          info_.set_progress(worker_->progress());
          info_.stop();
        }
        if (group_id_ == 0) {
          do_halt();
        } else {
          if (work_.rank() == 0) send_info(mcmp_tag::clone_info);
        }
        return;
      }
    }
    if (work_.rank() == 0) {
      info_.set_progress(worker_->progress());
      loops_ = timer_.next_loops(loops_);
    }
    broadcast(work_, loops_, 0);
    break;
  case mcmp_tag::clone_info :
    send_info(mcmp_tag::clone_info);
    break;
  case mcmp_tag::clone_checkpoint :
    do_checkpoint();
    send_info(mcmp_tag::clone_checkpoint);
    break;
  case mcmp_tag::clone_suspend :
    this->do_suspend();
    send_info(mcmp_tag::clone_suspend);
    break;
  case mcmp_tag::clone_halt :
    this->do_halt();
    send_halted();
    break;
  case mcmp_tag::do_nothing :
    break;
  default:
    if (work_.rank() == 0)
      std::cerr << "Warning: ignoring a message with an unknown tag " << tag << std::endl;
  }
}

void clone_mpi::checkpoint() {
  if (work_.rank() == 0) {
    int tag = mcmp_tag::clone_checkpoint;
    broadcast(work_, tag, 0);
    do_checkpoint();
  }
}

void clone_mpi::suspend() {
  if (work_.rank() == 0) {
    int tag = mcmp_tag::clone_suspend;
    broadcast(work_, tag, 0);
    do_suspend();
  }
}

void clone_mpi::do_checkpoint() {
  if (work_.rank() == 0 && info_.progress() < 1) info_.stop();
  this->save();
}

void clone_mpi::do_suspend() {
  if (work_.rank() == 0) info_.stop();
  this->save();
  work_.barrier();
  worker_.reset();
}

void clone_mpi::do_halt() {
  if (work_.rank() == 0 && info_.progress() < 1) {
    std::cerr << "clone is not finished\n";
    boost::throw_exception(std::logic_error("clone is not finished"));
  }
  this->save();
  work_.barrier();
  worker_.reset();
}

bool clone_mpi::halted() const { return !worker_; }

clone_info const& clone_mpi::info() const { return info_; }

void clone_mpi::load() {
  {
    IXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
    worker_->load_worker(dp);
  }
  #pragma omp critical (hdf5io)
  {
    hdf5::iarchive h5(complete(boost::filesystem::path(info_.dumpfile_h5()), basedir_));
    h5 >> make_pvp("/", this);
  }
}

void clone_mpi::save() const{
  {
    OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
    worker_->save_worker(dp);
  }
  #pragma omp critical (hdf5io)
  {
    hdf5::oarchive h5(complete(boost::filesystem::path(info_.dumpfile_h5()), basedir_));
    h5 << make_pvp("/", this);
  }
}

void clone_mpi::serialize(hdf5::iarchive& ar) {
  ar >> make_pvp("parameters", params_)
     >> make_pvp("log/alps", info_);
  load_observable(ar, clone_id_, work_.rank(), measurements_);
}

void clone_mpi::serialize(hdf5::oarchive& ar) const {
  ar << make_pvp("parameters", params_)
     << make_pvp("log/alps", info_);
  save_observable(ar, clone_id_, work_.rank(), measurements_);
}

void clone_mpi::output() const{
  std::cout << params_;
  BOOST_FOREACH(ObservableSet const& m, measurements_) std::cout << m;
}

void clone_mpi::send_info(mcmp_tag_t tag) {
  if (work_.rank() == 0) {
    if (ctrl_.rank() == 0) std::cerr << "Error: sending to myself in clone_mpi::send_info()\n";
    ctrl_.send(0, tag, clone_info_msg_t(task_id_, clone_id_, group_id_, info_));
  }
}

void clone_mpi::send_halted() {
  if (work_.rank() == 0) {
    if (ctrl_.rank() == 0) std::cerr << "Error: sending to myself in clone_mpi::send_halted()\n";
    ctrl_.send(0, mcmp_tag::clone_halt, clone_halt_msg_t(task_id_, clone_id_, group_id_));
  }
}

#endif // ALPS_HAVE_MPI

} // end namespace alps
