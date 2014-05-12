/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2012 by Synge Todo <wistaria@comp-phys.org>,
*                            Ryo Igarashi <rigarash@issp.u-tokyo.ac.jp>,
*                            Haruhiko Matsuo <halm@rist.or.jp>,
*                            Tatsuya Sakashita <t-sakashita@issp.u-tokyo.ac.jp>,
*                            Yuichi Motoyama <yomichi@looper.t.u-tokyo.ac.jp>
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

#include <alps/ngs/parapack/job.h>
#include <alps/ngs/parapack/clone.h>
#include <alps/parapack/filelock.h>
// #include "measurement.h"
#include <alps/ngs/parapack/simulation_p.h>

#include <alps/hdf5.hpp>

// some file (probably a python header) defines a tolower macro ...
#undef tolower
#undef toupper

#include <boost/config.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>

namespace alps {
namespace ngs_parapack {

//
// task
//

task::task() : status_(task_status::Undefined) {}

task::task(boost::filesystem::path const& file) : status_(task_status::Undefined) {
  basedir_ = file.branch_path();
  file_in_str_ = file.filename().string();
  file_out_str_ = regex_replace(file.filename().string(), boost::regex("\\.in\\.xml$"), ".out.xml");
  if (file_in_str_ == file_out_str_) {
    file_in_str_ = regex_replace(file.filename().string(), boost::regex("\\.out\\.xml$"), ".in.xml");
    file_out_str_ = file.filename().string();
  }
  base_ = regex_replace(file_out_str_, boost::regex("\\.out\\.xml$"), "");
}

bool task::on_memory() const {
  return status_ == task_status::Ready || status_ == task_status::Running ||
    status_ == task_status::Continuing || status_ == task_status::Idling;
}

task::range_type const& task::num_clones() const {
  if (!on_memory()) boost::throw_exception(std::logic_error(
    "task::num_clones() task not loaded"));
  return num_clones_;
}

uint32_t task::num_running() const {
  if (!on_memory()) boost::throw_exception(std::logic_error(
    "task::num_running() task not loaded"));
  return running_.size();
}

uint32_t task::num_suspended() const {
  if (!on_memory()) boost::throw_exception(std::logic_error(
    "task::num_suspended() task not loaded"));
  return suspended_.size();
}

uint32_t task::num_finished() const {
  if (!on_memory()) boost::throw_exception(std::logic_error(
    "task::num_finished() task not loaded"));
  return finished_.size();
}

uint32_t task::num_started() const {
  return num_running() + num_suspended() + num_finished();
}

void task::load() {
  if (on_memory()) boost::throw_exception(std::logic_error("task::load() task already loaded"));
  params_ = alps::params();
  clone_info_.clear();
  boost::filesystem::path file_in = complete(boost::filesystem::path(file_in_str_), basedir_);
  boost::filesystem::path file_out = complete(boost::filesystem::path(file_out_str_), basedir_);
  simulation_xml_handler handler(params_, clone_info_);
  XMLParser parser(handler);
  if (!exists(file_out)) {
    parser.parse(file_in);
  } else {
    filelock lock(file_out, /* lock_now = */ true, /* wait = */ 60);
    if (!lock.locked())
      boost::throw_exception(std::runtime_error("task::load() lock failed after 60 seconds"));
    parser.parse(file_out);
  }

  num_clones_ = (params_["NUM_CLONES"] | 1);
  clone_status_.clear();
  clone_master_.clear();
  running_.clear();
  suspended_.clear();
  finished_.clear();
  BOOST_FOREACH(clone_info const& info, clone_info_) {
    if (info.progress() < 1){
      clone_status_.push_back(clone_status::Suspended);
      suspended_.insert(info.clone_id());
    } else {
      clone_status_.push_back(clone_status::Finished);
      finished_.insert(info.clone_id());
    }
    clone_master_.push_back(Process());
  }

  status_ = task_status::Ready;
  progress_ = calc_progress();
  boost::tie(weight_, dump_weight_) = calc_weight();
  status_ = calc_status();
}

void task::save(bool write_xml) const {
  if (!on_memory()) boost::throw_exception(std::logic_error("task not loaded"));
  boost::filesystem::path file_out = complete(boost::filesystem::path(file_out_str_), basedir_);
  {
    filelock lock(file_out, /* lock_now = */ true, /* wait = */ 60);
    if (!lock.locked())
      boost::throw_exception(std::runtime_error("task::load() lock failed after 60 seconds"));
    if (exists(file_out)) {
      alps::params params_tmp;
      std::deque<clone_info> clone_info_tmp;
      simulation_xml_handler handler(params_tmp, clone_info_tmp);
      XMLParser parser(handler);
      parser.parse(file_out);
      simulation_xml_writer(file_out, write_xml, true, params_, clone_info_);
    } else {
      simulation_xml_writer(file_out, write_xml, true, params_, clone_info_);
    }
  }
}

void task::halt() {
  if (!on_memory()) boost::throw_exception(std::logic_error("task not loaded"));
  if (running_.size()) boost::throw_exception(std::logic_error("running clone exists"));

  switch (status_) {
  case task_status::Ready:
    status_ = task_status::NotStarted;
    break;
  case task_status::Running:
    status_ = task_status::Suspended;
    break;
  case task_status::Continuing:
    status_ = task_status::Finished;
    break;
  case task_status::Idling:
    status_ = task_status::Completed;
    break;
  default:
    boost::throw_exception(std::logic_error("unknown task_status"));
  }

  params_ = alps::params();
  clone_status_.clear();
  clone_master_.clear();
  clone_info_.clear();
  running_.clear();
  suspended_.clear();
  finished_.clear();
}

void task::check_parameter(bool write_xml) {
  // change in NUM_CLONES : keep old calculations and add new ones
  // change in SEED : ignored
  // change in other parameters : throw away all the old clones

  boost::filesystem::path file_in = complete(boost::filesystem::path(file_in_str_), basedir_);
  boost::filesystem::path file_out = complete(boost::filesystem::path(file_out_str_), basedir_);
  if (exists(file_out)) {
    // loading parameters from *.in.xml
    alps::params params_in;
    simulation_parameters_xml_handler handler(params_in);
    XMLParser parser(handler);
    parser.parse(file_in);

    // loading from *.out.xml
    load();
    if (params_.defined("SEED")) params_in["SEED"] = params_["SEED"];

    bool changed = false;
    for (alps::params::const_iterator p = params_.begin(); p != params_.end(); ++p) {
      if (p->first != "NUM_CLONES")
        if (!params_in.defined(p->first) ||
            boost::lexical_cast<std::string>(params_in[p->first]) != boost::lexical_cast<std::string>(p->second))
          changed = true;
    }
    for (alps::params::const_iterator p = params_in.begin(); p != params_in.end(); ++p) {
      if (p->first != "NUM_CLONES")
        if (!params_.defined(p->first)) changed = true;
    }
    if (changed) {
      std::cout << "Info: parameters in " << logger::task(task_id_) << " have been changed. "
                << "All the clones are being thrown away." << std::endl;
      params_ = params_in;
      clone_info_.clear();
      clone_status_.clear();
      clone_master_.clear();
      running_.clear();
      suspended_.clear();
      finished_.clear();
    } else if (params_in.defined("NUM_CLONES")) {
      if ((params_.defined("NUM_CLONES") &&
           (int(params_in["NUM_CLONES"]) != int(params_["NUM_CLONES"]))) ||
          !params_.defined("NUM_CLONES")) {
        std::cout << "Info: number of clones in " << logger::task(task_id_) << " has been changed."
                  << std::endl;
        params_["NUM_CLONES"] = params_in["NUM_CLONES"];
        changed = true;
      }
    }

    if (changed) {
      num_clones_ = (params_["NUM_CLONES"] | 1);
      progress_ = calc_progress();
      boost::tie(weight_, dump_weight_) = calc_weight();
      status_ = calc_status();
      save(write_xml);
    }
    halt();
  }
}

bool task::can_dispatch() const {
  return num_suspended() > 0 || num_started() < num_clones_.max BOOST_PREVENT_MACRO_SUBSTITUTION ();
}

void task::info_updated(cid_t cid, clone_info const& info) {
  if (clone_status_[cid] == clone_status::Running) {
    clone_info_[cid] = info;
    if (info.progress() >= 1) clone_status_[cid] = clone_status::Idling;
  }
}

void task::clone_halted(cid_t cid, clone_info const& info) {
  info_updated(cid, info);
  clone_halted(cid);
}

void task::clone_halted(cid_t cid) {
  if (clone_status_[cid] != clone_status::Stopping)
    boost::throw_exception(std::logic_error("clone is not stopping"));
  clone_status_[cid] = clone_status::Finished;
  running_.erase(cid);
  finished_.insert(cid);
  progress_ = calc_progress();
  status_ = calc_status();
  boost::tie(weight_, dump_weight_) = calc_weight();
}

void task::write_xml_summary(oxstream& os) const {
  os << start_tag("TASK")
     << attribute("id", task_id_+1)
     << attribute("status", task_status::to_string(status_))
     << attribute("progress", precision(progress() * 100, 3) + '%')
     << attribute("weight", precision(dump_weight_, 3))
     << start_tag("INPUT")
     << attribute("file", file_in_str_)
     << end_tag("INPUT")
     << start_tag("OUTPUT")
     << attribute("file", file_out_str_)
     << end_tag("OUTPUT")
     << end_tag("TASK");
}

void task::write_xml_archive(oxstream& os) const {
  os << alps::start_tag("SIMULATION")
     << attribute("id", task_id_+1)
     << attribute("status", task_status::to_string(status_))
     << attribute("progress", precision(progress() * 100, 3) + '%')
     << params_;
  for (unsigned int i = 0; i < clone_info_.size(); ++i) if (clone_info_[i].clone_id() == i) os << clone_info_[i];
  os << alps::end_tag("SIMULATION");
}

double task::calc_progress() const {
  if (!on_memory()) boost::throw_exception(std::logic_error("task not loaded"));
  return (double)(num_finished()) / num_clones().min BOOST_PREVENT_MACRO_SUBSTITUTION ();
}

std::pair<double, double> task::calc_weight() const {
  if (!on_memory()) boost::throw_exception(std::logic_error("task not loaded"));
  double w;
  if (num_suspended() > 0)
    w = 4.0;
  else if (num_started() == 0)
    // NotStarted
    w = 3.0;
  else if (num_started() < num_clones().min BOOST_PREVENT_MACRO_SUBSTITUTION ())
    // Running
    w = 2.0 - (double)(num_started()) / num_clones().min BOOST_PREVENT_MACRO_SUBSTITUTION ();
  else
    // Continuing
    w = 1.0 - (double)(num_started()) / num_clones().max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  double d = (num_running() + num_suspended() > 0) ? 4.0 : w;
  return std::make_pair(w, d);
}

task_status_t task::calc_status() const {
  if (!on_memory()) boost::throw_exception(std::logic_error("task not loaded"));
  if (num_started() == 0)
    return task_status::Ready;
  else if (num_finished() < num_clones_.min BOOST_PREVENT_MACRO_SUBSTITUTION ())
    return task_status::Running;
  else if (num_finished() < num_clones_.max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    return task_status::Continuing;
  else
    return task_status::Idling;
}

} // end namespace ngs_parapack
} // end namespace alps
