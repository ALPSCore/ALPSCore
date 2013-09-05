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

// boostinspect:nounnamed

#ifndef NGS_PARAPACK_FACTORY_H
#define NGS_PARAPACK_FACTORY_H

#include <alps/config.h>
#include <alps/hdf5.hpp>
#include <alps/ngs/params.hpp>
#include <boost/shared_ptr.hpp>

namespace alps {
namespace ngs_parapack {

//
// abstract_worker
//

class ALPS_DECL abstract_worker {
public:
  virtual ~abstract_worker();

  virtual bool run(boost::function<bool ()> const& stop_callback,
                   boost::function<void (double)> const& progress_callback) = 0;

  // the concrete subclass should implement the following 3 pure virtual functions
  virtual void load(alps::hdf5::archive& ar) = 0;
  virtual void save(alps::hdf5::archive& ar) const = 0;
  virtual double fraction_completed() const = 0;

  // these two are for internal use.  DO NOT OVERWRITE.
  virtual void load_worker(alps::hdf5::archive& ar);
  virtual void save_worker(alps::hdf5::archive& ar) const;
};


template<typename WORKER>
class ngs_worker : public abstract_worker {
public:
  ngs_worker(alps::params const& p) : worker_(p) {}
  virtual ~ngs_worker() {}

  virtual bool run(boost::function<bool ()> const& stop_callback,
                   boost::function<void (double)> const& progress_callback) 
  { return worker_.run(stop_callback, progress_callback); }

  // the concrete subclass should implement the following 3 pure virtual functions
  virtual void load(alps::hdf5::archive& ar) { worker_.load(ar); }
  virtual void save(alps::hdf5::archive& ar) const { worker_.save(ar); }
  virtual double fraction_completed() const { return worker_.fraction_completed(); }

  // these two are for internal use.  DO NOT OVERWRITE.
  virtual void load_worker(alps::hdf5::archive& ar) { this->load(ar); }
  virtual void save_worker(alps::hdf5::archive& ar) const { this->save(ar); }
private:
  WORKER worker_;
};


//
// creators
//

class abstract_worker_creator {
public:
  virtual ~abstract_worker_creator() {}
  virtual boost::shared_ptr<abstract_worker> create(alps::params const& p) const = 0;
};

template <typename WORKER>
class worker_creator : public abstract_worker_creator {
public:
  typedef ngs_worker<WORKER> worker_type;
  virtual ~worker_creator() {}
  boost::shared_ptr<abstract_worker> create(alps::params const& p) const {
    return boost::shared_ptr<abstract_worker>(new worker_type(p));
  }
};


//
// factory classes
//

class ALPS_DECL worker_factory : private boost::noncopyable {
private:
  typedef boost::shared_ptr<abstract_worker> worker_pointer_type;
  typedef boost::shared_ptr<abstract_worker_creator> creator_pointer_type;
public:
  static worker_pointer_type make_worker(alps::params const& p);
  template<typename WORKER>
  static void register_worker() {
    instance()->worker_creator_.reset(new worker_creator<WORKER>());
  }
  static worker_factory* instance();
private:
  static worker_factory* instance_;
  creator_pointer_type worker_creator_;
};

#ifdef ALPS_HAVE_MPI

template<typename WORKER>
class parallel_ngs_worker : public abstract_worker {
public:
  parallel_ngs_worker(boost::mpi::communicator const& comm, alps::params const& p) : worker_(comm, p) {}
  virtual ~parallel_ngs_worker() {}

  virtual bool run(boost::function<bool ()> const& stop_callback,
                   boost::function<void (double)> const& progress_callback) 
  { return worker_.run(stop_callback, progress_callback); }

  // the concrete subclass should implement the following 3 pure virtual functions
  virtual void load(alps::hdf5::archive& ar) { worker_.load(ar); }
  virtual void save(alps::hdf5::archive& ar) const { worker_.save(ar); }
  virtual double fraction_completed() const { return worker_.fraction_completed(); }

  // these two are for internal use.  DO NOT OVERWRITE.
  virtual void load_worker(alps::hdf5::archive& ar) { this->load(ar); }
  virtual void save_worker(alps::hdf5::archive& ar) const { this->save(ar); }
private:
  WORKER worker_;
};

//
// abstract_parallel_worker_creator
//

class abstract_parallel_worker_creator {
public:
  virtual ~abstract_parallel_worker_creator() {}
  virtual boost::shared_ptr<abstract_worker> create(boost::mpi::communicator const& comm,
    alps::params const& p) const = 0;
};

template <typename WORKER>
class parallel_worker_creator : public abstract_parallel_worker_creator {
public:
  typedef parallel_ngs_worker<WORKER> worker_type;
  virtual ~parallel_worker_creator() {}
  boost::shared_ptr<abstract_worker> create(boost::mpi::communicator const& comm,
    alps::params const& p) const {
    return boost::shared_ptr<abstract_worker>(new worker_type(comm, p));
  }
};


//
// parallel_worker_factory
//

class parallel_worker_factory : private boost::noncopyable {
private:
  typedef boost::shared_ptr<abstract_worker> worker_pointer_type;
  typedef boost::shared_ptr<abstract_parallel_worker_creator> creator_pointer_type;
public:
  static worker_pointer_type make_worker(boost::mpi::communicator const& comm,
    alps::params const& p);
  template<typename WORKER>
  static void register_worker() {
    instance()->worker_creator_.reset(new parallel_worker_creator<WORKER>());
  }
  static parallel_worker_factory* instance();
private:
  static parallel_worker_factory* instance_;
  creator_pointer_type worker_creator_;
};

#endif // ALPS_HAVE_MPI

} // end namespace ngs_parapack
} // end namespace alps

#endif // NGS_PARAPACK_FACTORY_H
