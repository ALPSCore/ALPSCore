/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2009 by Synge Todo <wistaria@comp-phys.org>
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

#include "parallel_factory.h"

namespace alps {
namespace parapack {

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
    std::cerr << "Error: no worker registered\n";
    boost::throw_exception(std::runtime_error("parallel_worker_factory::make_creator()"));
  }
  if (worker_creators_.size() == 1) {
    if (params.defined("WORKER") && worker_creators_.begin()->first != params["WORKER"]) {
      std::clog << "Warning: unknown worker: \"" << params["WORKER"]
                << "\".  The only worker \"" << worker_creators_.begin()->first
                << "\" will be used instead.\n";
    }
    return worker_creators_.begin()->second;
  }
  if (!params.defined("WORKER")) {
    std::cerr << "Error: no worker specified (registered workers: ";
    for (creator_map_type::const_iterator itr = worker_creators_.begin();
         itr != worker_creators_.end(); ++itr) {
      if (itr != worker_creators_.begin()) std::cerr << ", ";
      std::cerr << itr->first;
    }
    std::cerr << std::endl;
    boost::throw_exception(std::runtime_error("worker_factory::make_creator()"));
  }
  creator_map_type::const_iterator itr = worker_creators_.find(params["WORKER"]);
  if (itr == worker_creators_.end() || itr->second == 0) {
    std::cerr << "Error: unknown worker: \"" << params["WORKER"] << "\" (registered workers: ";
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

// initialization of static member pointer of factories
parallel_worker_factory* parallel_worker_factory::instance_ = 0;

} // end namespace parapack
} // end namespace alps
