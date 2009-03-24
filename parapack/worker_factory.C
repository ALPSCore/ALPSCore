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

#include "worker_factory.h"
#include "version.h"

namespace alps {
namespace parapack {

//
// abstract_worker
//

abstract_worker::~abstract_worker() {}

void abstract_worker::init_observables(Parameters const&, ObservableSet&) {
  boost::throw_exception(std::runtime_error("abstract_worker::init_observables() should be implemented"));
}
void abstract_worker::init_observables(Parameters const& params, std::vector<ObservableSet>& obs) {
  obs.resize(1);
  this->init_observables(params, obs[0]);
}

void abstract_worker::run(ObservableSet&) {
  boost::throw_exception(std::runtime_error("abstract_worker::run() should be implemented"));
}
void abstract_worker::run(std::vector<ObservableSet>& obs) {
  this->run(obs[0]);
}

void abstract_worker::load_worker(IDump& dump) { this->load(dump); }

void abstract_worker::save_worker(ODump& dump) const { this->save(dump); }

//
// abstract_evaluator
//

abstract_evaluator::~abstract_evaluator() {}

void abstract_evaluator::load(ObservableSet const&, ObservableSet&) {
  boost::throw_exception(std::runtime_error("abstract_evaluator::load() should be implemented"));
}

void abstract_evaluator::load(std::vector<ObservableSet> const& obs_in,
                              std::vector<ObservableSet>& obs_out) {
  if (obs_out.size() == 0) obs_out.resize(obs_in.size());
  if (obs_out.size() != obs_in.size())
    boost::throw_exception(std::runtime_error("inconsistent size of ObservableSet"));
  for (int i = 0; i < obs_out.size(); ++i) this->load(obs_in[i], obs_out[i]);
}

void abstract_evaluator::evaluate(ObservableSet&) const {
  boost::throw_exception(std::runtime_error("abstract_evaluator::evaluate() should be implemented"));
}

void abstract_evaluator::evaluate(std::vector<ObservableSet>& obs_out) const {
  for (int i = 0; i < obs_out.size(); ++i) this->evaluate(obs_out[i]);
}

//
// simple_evaluator
//

simple_evaluator::simple_evaluator() {}

simple_evaluator::simple_evaluator(Parameters const&) {}

simple_evaluator::~simple_evaluator() {}

void simple_evaluator::load(ObservableSet const& obs_in, ObservableSet& obs_out) {
  obs_out << obs_in;
}

void simple_evaluator::evaluate(ObservableSet&) const {}


//
// worker_factory
//

worker_factory::worker_factory() : copyright_string_(), version_string_() {}

void worker_factory::print_copyright(std::ostream& out) {
  if (instance()->copyright_string_.size())
    out << instance()->copyright_string_;
  else if (instance()->version_string_.size())
    out << instance()->version_string_ << std::endl;
  else
    out << PARAPACK_COPYRIGHT << std::endl;
}

std::string worker_factory::version() {
  if (instance()->version_string_.size())
    return instance()->version_string_;
  else
    return PARAPACK_VERSION_STRING;
}

bool worker_factory::set_copyright(std::string const& str) {
  copyright_string_ = str;
  return true;
}

bool worker_factory::set_version(std::string const& str) {
  version_string_ = str;
  return true;
}

worker_factory::worker_pointer_type worker_factory::make_worker(Parameters const& params) {
  return instance()->make_creator(params)->create(params);
}

alps::scheduler::MCSimulation* worker_factory::make_task(alps::ProcessList const& nodes,
  boost::filesystem::path const& file) {
  return new alps::scheduler::MCSimulation(nodes, file);
}

alps::scheduler::MCSimulation* worker_factory::make_task(alps::ProcessList const& nodes,
  boost::filesystem::path const& file, alps::Parameters const&) {
  return new alps::scheduler::MCSimulation(nodes, file);
}

alps::scheduler::MCSimulation* worker_factory::make_task(alps::ProcessList const&,
  alps::Parameters const&) {
  return 0;
}

alps::scheduler::MCRun* worker_factory::make_worker(alps::ProcessList const&,
  alps::Parameters const& params, int) {
  return instance()->make_creator(params)->create_alps_worker(params);
}

bool worker_factory::unregister_worker(std::string const& name) {
  creator_map_type::iterator itr = worker_creators_.find(name);
  if (itr == worker_creators_.end()) return false;
  worker_creators_.erase(itr);
  return true;
}

worker_factory* worker_factory::instance() {
  if (!instance_) instance_ = new worker_factory;
  return instance_;
}

worker_factory::creator_pointer_type worker_factory::make_creator(Parameters const& params) const {
  if (worker_creators_.size() == 0) {
    std::cerr << "No worker registered\n";
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  if (!params.defined("WORKER")) {
    if (worker_creators_.size() == 1) {
      return worker_creators_.begin()->second;
    } else {
      std::cerr << "Please specify one of the workers (";
      for (creator_map_type::const_iterator itr = worker_creators_.begin();
           itr != worker_creators_.end(); ++itr) {
        if (itr != worker_creators_.begin()) std::cerr << ", ";
        std::cerr << itr->first;
      }
      std::cerr << ") by WORKER parameter\n";
      boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
    }
  }
  creator_map_type::const_iterator itr = worker_creators_.find(params["WORKER"]);
  if (itr == worker_creators_.end() || itr->second == 0) {
    std::cerr << "Unknown worker: " << params["WORKER"] << " (registered workers: ";
    for (creator_map_type::const_iterator itr = worker_creators_.begin();
         itr != worker_creators_.end(); ++itr) {
      if (itr != worker_creators_.begin()) std::cerr << ", ";
      std::cerr << itr->first;
    }
    std::cerr << ")\n";
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  return itr->second;
}


//
// evaluator_factory
//

evaluator_factory::evaluator_pointer_type
evaluator_factory::make_evaluator(Parameters const& params) {
  return instance()->make_creator(params)->create(params);
}

bool evaluator_factory::unregister_evaluator(std::string const& name) {
  creator_map_type::iterator itr = evaluator_creators_.find(name);
  if (itr == evaluator_creators_.end()) return false;
  evaluator_creators_.erase(itr);
  return true;
}

evaluator_factory* evaluator_factory::instance() {
  if (!instance_) instance_ = new evaluator_factory;
  return instance_;
}

evaluator_factory::creator_pointer_type
evaluator_factory::make_creator(Parameters const& params) const {
  if (params.defined("EVALUATOR")) {
    if (params["EVALUATOR"] == "defualt") {
      return creator_pointer_type(new evaluator_creator<simple_evaluator>);
    } else {
      creator_map_type::const_iterator itr = evaluator_creators_.find(params["EVALUATOR"]);
      if (itr == evaluator_creators_.end() || itr->second == 0) {
        std::cerr << "Unknown evaluator: " << params["EVALUATOR"] << " (registered evaluators: ";
        for (creator_map_type::const_iterator itr = evaluator_creators_.begin();
             itr != evaluator_creators_.end(); ++itr) {
          if (itr != evaluator_creators_.begin()) std::cerr << ", ";
          std::cerr << itr->first;
        }
        std::cerr << ")\n";
        boost::throw_exception(std::runtime_error("evaluator_factory::make_creator()"));
      }
      return itr->second;
    }
  } else if (params.defined("WORKER")) {
    creator_map_type::const_iterator itr = evaluator_creators_.find(params["WORKER"]);
    if (itr == evaluator_creators_.end() || itr->second == 0) {
      std::clog << "Info: unknown evaluator: " << params["WORKER"]
                << ".  Using default evaluator.\n";
      return creator_pointer_type(new evaluator_creator<simple_evaluator>);
    } else {
      return itr->second;
    }
  }
  // default evaluator
  return creator_pointer_type(new evaluator_creator<simple_evaluator>);
}

//
// initialization of static member pointer of factories
//

worker_factory* worker_factory::instance_ = 0;

evaluator_factory* evaluator_factory::instance_ = 0;

} // end namespace parapack
} // end namespace alps
