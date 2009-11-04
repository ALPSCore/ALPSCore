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

#ifndef PARAPACK_PARALLEL_FACTORY_H
#define PARAPACK_PARALLEL_FACTORY_H

#include "worker_factory.h"
#include "process_mpi.h"
#include <alps/parameter.h>
#include <boost/shared_ptr.hpp>

namespace alps {
namespace parapack {

//
// creator
//

class abstract_parallel_worker_creator {
public:
  virtual ~abstract_parallel_worker_creator() {}
  virtual boost::shared_ptr<abstract_worker> create(boost::mpi::communicator const& comm,
    const Parameters& params) const = 0;
};

template <typename WORKER>
class parallel_worker_creator : public abstract_parallel_worker_creator {
public:
  typedef WORKER worker_type;
  virtual ~parallel_worker_creator() {}
  boost::shared_ptr<abstract_worker> create(boost::mpi::communicator const& comm,
    Parameters const& params) const {
    return boost::shared_ptr<abstract_worker>(new worker_type(comm, params));
  }
  void print_copyright(std::ostream& out) const { worker_type::print_copyright(out); }
  std::string version() const { return worker_type::version(); }
};


//
// factory
//

class parallel_worker_factory : private boost::noncopyable {
private:
  typedef boost::shared_ptr<abstract_worker> worker_pointer_type;
  typedef boost::shared_ptr<abstract_parallel_worker_creator> creator_pointer_type;
  typedef std::map<std::string, creator_pointer_type> creator_map_type;

public:
  static worker_pointer_type make_worker(boost::mpi::communicator const& comm,
    Parameters const& params);

  template<typename WORKER>
  bool register_worker(std::string const& name) {
    bool isnew = (worker_creators_.find(name) == worker_creators_.end());
    worker_creators_[name] = creator_pointer_type(new parallel_worker_creator<WORKER>());
    return isnew;
  }
  bool unregister_worker(std::string const& name);

  static parallel_worker_factory* instance();

protected:
  creator_pointer_type make_creator(Parameters const& params) const;

private:
  static parallel_worker_factory* instance_;
  creator_map_type worker_creators_;
};

} // end namespace parapack
} // end namespace alps

#define PARAPACK_REGISTER_PARALLEL_ALGORITHM(worker, name) \
namespace { \
  const bool BOOST_JOIN(worker_, __LINE__) \
    = alps::parapack::parallel_worker_factory::instance()->register_worker<worker>(name); \
}

#define PARAPACK_REGISTER_PARALLEL_WORKER(worker, name) \
namespace { \
  const bool BOOST_JOIN(worker_, __LINE__) \
    = alps::parapack::parallel_worker_factory::instance()->register_worker<worker>(name); \
}

#endif // PARAPACK_PARALLEL_FACTORY_H
