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

// boostinspect:nounnamed

#ifndef PARAPACK_FACTORY_H
#define PARAPACK_FACTORY_H

#include "process.h"
#include <alps/config.h>
#include <alps/alea.h>
#include <alps/parameter.h>
#include <alps/scheduler.h>
#include <boost/shared_ptr.hpp>

namespace alps {
namespace parapack {

//
// abstract_worker
//

class ALPS_DECL abstract_worker {
public:
  virtual ~abstract_worker();

  // one of the following virtual member functions should be overwritten by the subclass
  virtual void init_observables(Parameters const& params, ObservableSet& obs);
  virtual void init_observables(Parameters const& params, std::vector<ObservableSet>& obs);

  // one of the following virtual member functions should be overwritten by the subclass
  virtual void run(ObservableSet& obs);
  virtual void run(std::vector<ObservableSet>& obs);

  // the concrete subclass should implement the following 4 pure virtual functions
  virtual void load(IDump& dp) = 0;
  virtual void save(ODump& dp) const = 0;
  virtual bool is_thermalized() const = 0;
  virtual double progress() const = 0;

  // these two are for internal use.  DO NOT OVERWRITE.
  virtual void load_worker(IDump& dp);
  virtual void save_worker(ODump& dp) const;
};


//
// alps_worker : wrapper used for conventional alps scheduler
//

template<typename WORKER>
class alps_worker : public alps::scheduler::MCRun {
public:
  alps_worker(alps::ProcessList const& w, alps::Parameters const& p, int n) :
    alps::scheduler::MCRun(w, p, n),  worker_(new WORKER(p)) {
    worker_->init_observables(p, measurements);
  }
  virtual ~alps_worker() {}
  virtual void dostep() { worker_->run(measurements); }
  virtual void load(alps::IDump& dump) { worker_->load(dump); }
  virtual void save(alps::ODump& dump) const { worker_->save(dump); }
  virtual bool is_thermalized() const { return worker_->is_thermalized(); }
  virtual double work_done() const { return worker_->progress(); }
private:
  boost::shared_ptr<abstract_worker> worker_;
};


//
// abstract_evaluator & simple_evaluator
//

class ALPS_DECL abstract_evaluator {
public:
  virtual ~abstract_evaluator();

  // one of the following virtual member functions should be overwritten by the subclass
  virtual void load(ObservableSet const& obs_in, ObservableSet& obs_out);
  virtual void load(std::vector<ObservableSet> const& obs_in, std::vector<ObservableSet>& obs_out);

  // one of the following virtual member functions should be overwritten by the subclass
  virtual void evaluate(ObservableSet& obs_out) const;
  virtual void evaluate(std::vector<ObservableSet>& obs_out) const;
};

class ALPS_DECL simple_evaluator : public abstract_evaluator {
public:
  simple_evaluator();
  simple_evaluator(Parameters const& params);
  virtual ~simple_evaluator();
  virtual void load(ObservableSet const& obs_in, ObservableSet& obs_out);
  virtual void evaluate(ObservableSet& obs_out) const;
};


//
// creators
//

class abstract_worker_creator {
public:
  virtual ~abstract_worker_creator() {}
  virtual boost::shared_ptr<abstract_worker> create(const Parameters& params) const = 0;
  virtual alps::scheduler::MCRun* create_alps_worker(const Parameters& params) const = 0;
};

template <typename WORKER>
class worker_creator : public abstract_worker_creator {
public:
  typedef WORKER worker_type;
  typedef alps_worker<WORKER> alps_worker_type;
  virtual ~worker_creator() {}
  boost::shared_ptr<abstract_worker> create(Parameters const& params) const {
    return boost::shared_ptr<abstract_worker>(new worker_type(params));
  }
  alps::scheduler::MCRun* create_alps_worker(const Parameters& params) const {
    return new alps_worker_type(alps::ProcessList(), params, 0);
  }
};

class abstract_evaluator_creator {
public:
  virtual ~abstract_evaluator_creator() {}
  virtual boost::shared_ptr<abstract_evaluator> create(Parameters const& params) const = 0;
};

template <typename EVALUATOR>
class evaluator_creator : public abstract_evaluator_creator {
public:
  typedef EVALUATOR evaluator_type;
  virtual ~evaluator_creator() {}
  boost::shared_ptr<abstract_evaluator> create(Parameters const& params) const {
    return boost::shared_ptr<abstract_evaluator>(new evaluator_type(params));
  }
};


//
// factory classes
//

class ALPS_DECL worker_factory : private boost::noncopyable {
private:
  typedef boost::shared_ptr<abstract_worker> worker_pointer_type;
  typedef boost::shared_ptr<abstract_worker_creator> creator_pointer_type;
  typedef std::map<std::string, creator_pointer_type> creator_map_type;

public:
  worker_factory();

  static void print_copyright(std::ostream& out);
  static std::string version();
  bool set_copyright(std::string const& str);
  bool set_version(std::string const& str);

  static worker_pointer_type make_worker(Parameters const& params);

  static alps::scheduler::MCSimulation* make_task(alps::ProcessList const& nodes,
    boost::filesystem::path const& file);
  static alps::scheduler::MCSimulation* make_task(alps::ProcessList const& nodes,
    boost::filesystem::path const& file, alps::Parameters const& params);
  static alps::scheduler::MCSimulation* make_task(alps::ProcessList const& nodes,
    alps::Parameters const& params);
  static alps::scheduler::MCRun* make_worker(alps::ProcessList const& nodes,
    alps::Parameters const& params, int n);

  template<typename WORKER>
  bool register_worker(std::string const& name) {
    bool isnew = (worker_creators_.find(name) == worker_creators_.end());
    worker_creators_[name] = creator_pointer_type(new worker_creator<WORKER>());
    return isnew;
  }
  bool unregister_worker(std::string const& name);

  static worker_factory* instance();

protected:
  creator_pointer_type make_creator(Parameters const& params) const;

private:
  static worker_factory* instance_;
  std::string copyright_string_;
  std::string version_string_;
  creator_map_type worker_creators_;
};


class ALPS_DECL evaluator_factory : private boost::noncopyable {
private:
  typedef boost::shared_ptr<abstract_evaluator> evaluator_pointer_type;
  typedef boost::shared_ptr<abstract_evaluator_creator> creator_pointer_type;
  typedef std::map<std::string, creator_pointer_type> creator_map_type;

public:
  static evaluator_pointer_type make_evaluator(Parameters const& params);
  template<typename EVALUATOR>
  bool register_evaluator(std::string const& name) {
    bool isnew = (evaluator_creators_.find(name) == evaluator_creators_.end());
    evaluator_creators_[name] = creator_pointer_type(new evaluator_creator<EVALUATOR>());
    return isnew;
  }
  bool unregister_evaluator(std::string const& name);
  static evaluator_factory* instance();

protected:
  creator_pointer_type make_creator(Parameters const& params) const;

private:
  static evaluator_factory* instance_;
  creator_map_type evaluator_creators_;
};

#ifdef ALPS_HAVE_MPI

//
// abstract_parallel_worker_creator
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
// parallel_worker_factory
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

#endif // ALPS_HAVE_MPI

} // end namespace parapack
} // end namespace alps

#define PARAPACK_SET_COPYRIGHT(str) \
namespace { \
const bool BOOST_JOIN(copyright_, __LINE__) \
  = alps::parapack::worker_factory::instance()->set_copyright(str); \
}

#define PARAPACK_SET_VERSION(str) \
namespace { \
const bool BOOST_JOIN(version_, __LINE__) \
  = alps::parapack::worker_factory::instance()->set_version(str); \
}

#define PARAPACK_REGISTER_ALGORITHM(worker, name) \
namespace { \
const bool BOOST_JOIN(worker_, __LINE__) \
  = alps::parapack::worker_factory::instance()->register_worker<worker>(name); \
}

#define PARAPACK_REGISTER_WORKER(worker, name) \
namespace { \
const bool BOOST_JOIN(worker_, __LINE__) \
  = alps::parapack::worker_factory::instance()->register_worker<worker>(name); \
}

#define PARAPACK_REGISTER_EVALUATOR(evaluator, name) \
namespace { \
const bool BOOST_JOIN(evaluator_, __LINE__) \
  = alps::parapack::evaluator_factory::instance()->register_evaluator<evaluator>(name); \
}

#ifdef ALPS_HAVE_MPI

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

#endif // ALPS_HAVE_MPI

#endif // PARAPACK_FACTORY_H
