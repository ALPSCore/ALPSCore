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

#include "clone.h"
#include <boost/filesystem/operations.hpp>

namespace alps {

bool load_observable(IDump& dp, Parameters& params, clone_info& info,
  std::vector<ObservableSet>& obs) {
  int32_t tag;
  dp >> tag;
  if (tag != parapack_dump::run_master && tag != parapack_dump::run_slave) {
    std::cerr << "dump does not contain a master/slave clone\n";
    return false;
  }

  int32_t version, np, rank;
  dp >> version >> np >> rank;
  dp.set_version(version);

  if (version < parapack_dump::initial_version || version > parapack_dump::current_version) {
    std::cerr << "The clone on dump is version " << version
              << " but this program can read only versions between "
              << parapack_dump::initial_version << " and " << parapack_dump::current_version
              << std::endl;
    return false;
  }

  if (version >= 304) {
    dp >> params >> info >> obs;
  } else {
    obs.resize(1);
    dp >> params >> info >> obs[0];
  }
  return true;
}

bool load_observable(boost::filesystem::path const& file, Parameters& params, clone_info& info,
  std::vector<ObservableSet>& obs) {
  if (!exists(file)) return false;
  IXDRFileDump dp(file);
  return load_observable(dp, params, info, obs);
}

bool load_observable(IDump& dp, std::vector<ObservableSet>& obs) {
  Parameters params;
  clone_info info;
  return load_observable(dp, params, info, obs);
}

bool load_observable(boost::filesystem::path const& file, std::vector<ObservableSet>& obs) {
  if (!exists(file)) return false;
  IXDRFileDump dp(file);
  return load_observable(dp, obs);
}

clone::clone(tid_t tid, cid_t cid, Parameters const& params,
  boost::filesystem::path const& basedir, std::string const& base, bool is_new)
  : task_id_(tid), clone_id_(cid), params_(params), basedir_(basedir) {
  params_["DIR_NAME"] = basedir_.native_file_string();
  params_["BASE_NAME"] = base;

  info_ = clone_info(clone_id_, params_, base);
  params_["WORKER_SEED"] = info_.worker_seed();
  params_["DISORDER_SEED"] = info_.disorder_seed();

  worker_ = parapack::worker_factory::make_worker(params_);
  if (is_new) worker_->init_observables(params_, measurements_);
  if (!is_new) {
    IXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
    this->load(dp);
  }

  if (is_new && worker_->is_thermalized()) // no thermalization steps
    BOOST_FOREACH(alps::ObservableSet& m, measurements_) m.reset(true);

  if (is_new || worker_->progress() < 1)
    info_.start((worker_->is_thermalized()) ? "running" : "equilibrating");
  if (is_new && worker_->progress() >= 1) {
    info_.set_progress(worker_->progress());
    info_.stop();
    halt();
  }
}

clone::~clone() {}

void clone::run() {
  bool thermalized = worker_->is_thermalized();
  double progress = worker_->progress();

  worker_->run(measurements_);

  info_.set_progress(worker_->progress());
  if (!thermalized && worker_->is_thermalized()) {
    info_.stop();
    BOOST_FOREACH(alps::ObservableSet& m, measurements_) m.reset(true);
    info_.start("running");
  }
  if (progress < 1 && info_.progress() >= 1) {
    info_.stop();
    halt();
  }
}

bool clone::halted() const { return !worker_; }

double clone::progress() const { return info_.progress(); }

clone_info const& clone::info() const { return info_; }

void clone::load(IDump& dp) {
  load_observable(dp, params_, info_, measurements_);
  bool full_dump(dp);
  if (full_dump) worker_->load_worker(dp);
}

void clone::save(ODump& dp) const{
  dp << static_cast<int32_t>(parapack_dump::run_master)
     << static_cast<int32_t>(parapack_dump::current_version)
     << static_cast<int32_t>(1)
     << static_cast<int32_t>(0);

  dp << params_ << info_ << measurements_;

  bool full_dump = false;
  full_dump =
    (info_.progress() < 1) || params_.value_or_default("SCHEDULER_KEEP_FULL_DUMP", false);
  dp << full_dump;
  if (full_dump) worker_->save_worker(dp);
}

void clone::checkpoint() {
  if (info_.progress() < 1) info_.stop();
  OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
  this->save(dp);
}

void clone::suspend() {
  info_.stop();
  OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
  this->save(dp);
  worker_.reset();
}

void clone::halt() {
  if (info_.progress() < 1)
    boost::throw_exception(std::logic_error("clone is not finished"));
  OXDRFileDump dp(complete(boost::filesystem::path(info_.dumpfile()), basedir_));
  this->save(dp);
  worker_.reset();
}

void clone::output() const{
  std::cout << params_;
  BOOST_FOREACH(ObservableSet const& m, measurements_) std::cout << m;
}

} // end namespace alps
