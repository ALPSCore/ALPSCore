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
  for (std::size_t r = 0; r < obs_out.size(); ++r) this->load(obs_in[r], obs_out[r]);
}

void abstract_evaluator::evaluate(ObservableSet&) const {
  boost::throw_exception(std::runtime_error("abstract_evaluator::evaluate() should be implemented"));
}

void abstract_evaluator::evaluate(std::vector<ObservableSet>& obs_out) const {
  BOOST_FOREACH(ObservableSet& m, obs_out) { this->evaluate(m); }
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
    out << parapack_copyright() << std::endl;
}

std::string worker_factory::version() {
  if (instance()->version_string_.size())
    return instance()->version_string_;
  else
    return "ALPS/parapack scheduler";
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
    std::cerr << "Error: no algorithm registered\n";
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  std::string algoname = "";
  if (params.defined("ALGORITHM")) {
    algoname = params["ALGORITHM"];
  } else if (params.defined("WORKER")) {
    algoname = params["WORKER"];
    std::cout << "Warning: parameter WORKER is obsolete.  Please use ALGORITHM instead.\n";
  }
  if (worker_creators_.size() == 1) {
    if (algoname != "" && worker_creators_.begin()->first != algoname) {
      std::cout << "Warning: unknown algorithm: \"" << algoname
                << "\".  The only algorithm \"" << worker_creators_.begin()->first
                << "\" will be used instead.\n";
    }
    return worker_creators_.begin()->second;
  }
  if (algoname == "") {
    std::cerr << "Error: no algorithm specified (registered algorithms: ";
    for (creator_map_type::const_iterator itr = worker_creators_.begin();
         itr != worker_creators_.end(); ++itr) {
      if (itr != worker_creators_.begin()) std::cerr << ", ";
      std::cerr << "\"" << itr->first << "\"";
    }
    std::cerr << std::endl;
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  creator_map_type::const_iterator itr = worker_creators_.find(algoname);
  if (itr == worker_creators_.end() || itr->second == 0) {
    std::cerr << "Error: unknown algorithm: \"" << algoname << "\" (registered algorithms: ";
    for (creator_map_type::const_iterator itr = worker_creators_.begin();
         itr != worker_creators_.end(); ++itr) {
      if (itr != worker_creators_.begin()) std::cerr << ", ";
      std::cerr << "\"" << itr->first << "\"";
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
  std::string evalname = "";
  if (params.defined("EVALUATOR")) evalname = params["EVALUATOR"];
  std::string algoname = "";
  if (evalname == "") {
    if (params.defined("ALGORITHM")) {
      algoname = params["ALGORITHM"];
    } else if (params.defined("WORKER")) {
      algoname = params["WORKER"];
      std::cout << "Warning: parameter WORKER is obsolete.  Please use ALGORITHM instead.\n";
    }
  }
  if (evalname == "default") {
    // return default evaluator
  } else if (evaluator_creators_.size() == 0) {
    if (evalname != "") {
      std::cout << "Warning: unknown evaluator: " << evalname
                << ".  The default evaluator will be used instead\n";
    } else if (algoname != "") {
      std::cout << "Warning: unknown evaluator: " << algoname
                << ".  The default evaluator will be used instead\n";
    } else {
      std::cout << "Info: no evaluator registered.  The default evaluator will be used";
    }
  } else if (evaluator_creators_.size() == 1) {
    if (evalname != "" && evaluator_creators_.begin()->first != evalname) {
      std::cout << "Warning: unknown evaluator: \"" << evalname
                << "\".  The only evaluator \"" << evaluator_creators_.begin()->first
                << "\" will be used instead.\n";
    } else if (algoname != "" && evaluator_creators_.begin()->first != algoname) {
      std::cout << "Warning: unknown evaluator: \"" << algoname
                << "\".  The only evaluator \"" << evaluator_creators_.begin()->first
                << "\" will be used instead.\n";
    }
    return evaluator_creators_.begin()->second;
  } else if (evalname != "") {
    creator_map_type::const_iterator itr = evaluator_creators_.find(evalname);
    if (itr != evaluator_creators_.end() && itr->second != 0) {
      return itr->second;
    } else {
      std::cout << "Warning: unknown evaluator: \"" << evalname
                << "\" (registered evaluators: ";
      for (creator_map_type::const_iterator itr = evaluator_creators_.begin();
           itr != evaluator_creators_.end(); ++itr) {
        if (itr != evaluator_creators_.begin()) std::cout << ", ";
        std::cout << "\"" << itr->first << "\"";
      }
      std::cout << ").  The default evaluator will be used instead.\n";
    }
  } else if (algoname != "") {
    creator_map_type::const_iterator itr = evaluator_creators_.find(algoname);
    if (itr != evaluator_creators_.end() && itr->second != 0) {
      return itr->second;
    } else {
      std::cout << "Warning: unknown evaluator: \"" << algoname
                << "\" (registered evaluators: ";
      for (creator_map_type::const_iterator itr = evaluator_creators_.begin();
           itr != evaluator_creators_.end(); ++itr) {
        if (itr != evaluator_creators_.begin()) std::cout << ", ";
        std::cout << "\"" << itr->first << "\"";
      }
      std::cout << ").  The default evaluator will be used instead.\n";
    }
  }
  // default evaluator
  return creator_pointer_type(new evaluator_creator<simple_evaluator>);
}

#ifdef ALPS_HAVE_MPI

//
// parallel_worker_factory
//

parallel_worker_factory::worker_pointer_type
parallel_worker_factory::make_worker(boost::mpi::communicator const& comm,
  Parameters const& params) {
  return instance()->make_creator(params)->create(comm, params);
}

bool parallel_worker_factory::unregister_worker(std::string const& name) {
  creator_map_type::iterator itr = worker_creators_.find(name);
  if (itr == worker_creators_.end()) return false;
  worker_creators_.erase(itr);
  return true;
}

parallel_worker_factory* parallel_worker_factory::instance() {
  if (!instance_) instance_ = new parallel_worker_factory;
  return instance_;
}

parallel_worker_factory::creator_pointer_type
parallel_worker_factory::make_creator(Parameters const& params) const {
  if (worker_creators_.size() == 0) {
    std::cerr << "Error: no algorithm registered\n";
    boost::throw_exception(std::runtime_error("parallel_worker_factory::make_creator()"));
  }
  std::string algoname = "";
  if (params.defined("ALGORITHM")) {
    algoname = params["ALGORITHM"];
  } else if (params.defined("WORKER")) {
    algoname = params["WORKER"];
    std::cout << "Warning: parameter WORKER is obsolete.  Please use ALGORITHM instead.\n";
  }
  if (worker_creators_.size() == 1) {
    if (algoname != "" && worker_creators_.begin()->first != algoname) {
      std::cout << "Warning: unknown algorithm: \"" << algoname
                << "\".  The only algorithm \"" << worker_creators_.begin()->first
                << "\" will be used instead.\n";
    }
    return worker_creators_.begin()->second;
  }
  if (algoname == "") {
    std::cerr << "Error: no algorithm specified (registered algorithms: ";
    for (creator_map_type::const_iterator itr = worker_creators_.begin();
         itr != worker_creators_.end(); ++itr) {
      if (itr != worker_creators_.begin()) std::cerr << ", ";
      std::cerr << "\"" << itr->first << "\"";
    }
    std::cerr << std::endl;
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  creator_map_type::const_iterator itr = worker_creators_.find(algoname);
  if (itr == worker_creators_.end() || itr->second == 0) {
    std::cerr << "Error: unknown algorithm: \"" << algoname << "\" (registered algorithms: ";
    for (creator_map_type::const_iterator itr = worker_creators_.begin();
         itr != worker_creators_.end(); ++itr) {
      if (itr != worker_creators_.begin()) std::cerr << ", ";
      std::cerr << "\"" << itr->first << "\"";
    }
    std::cerr << ")\n";
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  return itr->second;
}

#endif // ALPS_HAVE_MPI

//
// initialization of static member pointer of factories
//

worker_factory* worker_factory::instance_ = 0;

evaluator_factory* evaluator_factory::instance_ = 0;

#ifdef ALPS_HAVE_MPI

parallel_worker_factory* parallel_worker_factory::instance_ = 0;

#endif // ALPS_HAVE_MPI

} // end namespace parapack
} // end namespace alps
