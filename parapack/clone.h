/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_CLONE_H
#define PARAPACK_CLONE_H

#include "clone_info.h"
#include "clone_timer.h"
#include "option.h"
#include "types.h"
#include "worker_factory.h"

#include <alps/alea.h>
#include <alps/hdf5.hpp>
#include <alps/osiris.h>
#include <alps/parameter.h>
#include <alps/scheduler/info.h>

namespace alps {

ALPS_DECL void save_observable(hdf5::archive & ar, std::vector<ObservableSet> const& obs);
ALPS_DECL void save_observable(hdf5::archive & ar, cid_t cid,
  std::vector<ObservableSet> const& obs);
ALPS_DECL void save_observable(hdf5::archive& ar, cid_t cid, int rank,
  std::vector<ObservableSet> const& obs);
ALPS_DECL bool load_observable(hdf5::archive & ar, std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(hdf5::archive & ar, std::string const& prefix, cid_t cid,
  std::string const& suffix, std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(hdf5::archive & ar, cid_t cid,
  std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(hdf5::archive & ar, cid_t cid, int rank,
  std::vector<ObservableSet>& obs);
ALPS_DECL bool load_observable(alps::IDump& dp, std::vector<ObservableSet>& obs);

class abstract_clone : public boost::noncopyable {
public:
  virtual ~abstract_clone() {}
  virtual void run() = 0;
  virtual bool halted() const = 0;
  virtual clone_info const& info() const = 0;

  virtual void load() = 0;
  virtual void save() const = 0;
  virtual void load(hdf5::archive &) = 0;
  virtual void save(hdf5::archive &) const = 0;

  virtual void checkpoint() = 0;
  virtual void suspend() = 0;
};

class clone : public abstract_clone {
public:
  clone(boost::filesystem::path const& basedir, alps::parapack::option opt, tid_t tid, cid_t cid,
    Parameters const& params, std::string const& base, bool is_new);
  virtual ~clone();

  tid_t task_id() const { return task_id_; }
  cid_t clone_id() const { return clone_id_; }

  void run();
  bool halted() const;
  clone_info const& info() const;

  void load();
  void save() const;
  void load(hdf5::archive & ar);
  void save(hdf5::archive & ar) const;

  void checkpoint();
  void suspend();

  void output() const;

protected:
  void do_halt();

private:
  tid_t task_id_;
  cid_t clone_id_;

  Parameters params_;

  boost::filesystem::path basedir_;
  clone_info info_;
  std::vector<ObservableSet> measurements_;

  dump_format_t dump_format_;
  dump_policy_t dump_policy_;
  clone_timer timer_;
  clone_timer::loops_t loops_;

  boost::shared_ptr<parapack::abstract_worker> worker_;
};

#ifdef ALPS_HAVE_MPI

struct clone_create_msg_t {
  clone_create_msg_t() {}
  clone_create_msg_t(tid_t tid, cid_t cid, gid_t gid, Parameters const& p, std::string const& bs,
    bool in) : task_id(tid), clone_id(cid), group_id(gid), params(p), base(bs), is_new(in) {}
  tid_t task_id;
  cid_t clone_id;
  gid_t group_id;
  Parameters params;
  std::string base;
  bool is_new;
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  template<class Archive>
  void save(Archive & ar, const unsigned int) const {
    ar & task_id & clone_id & group_id & params & base & is_new;
  }
  template<class Archive>
  void load(Archive & ar, const unsigned int) {
    ar & task_id & clone_id & group_id & params & base & is_new;
  }
};

struct clone_info_msg_t {
  clone_info_msg_t() {}
  clone_info_msg_t(tid_t tid, cid_t cid, gid_t gid, clone_info const& f)
    : task_id(tid), clone_id(cid), group_id(gid), info(f) {}
  tid_t task_id;
  cid_t clone_id;
  gid_t group_id;
  clone_info info;
  template<typename Archive>
  void serialize(Archive & ar, const unsigned int) { ar & task_id & clone_id & group_id & info; }
};

struct clone_halt_msg_t {
  clone_halt_msg_t() {}
  clone_halt_msg_t(tid_t tid, cid_t cid, gid_t gid) : task_id(tid), clone_id(cid), group_id(gid) {}
  tid_t task_id;
  cid_t clone_id;
  gid_t group_id;
  template<typename Archive>
  void serialize(Archive & ar, const unsigned int) { ar & task_id & clone_id & group_id; }
};

class clone_mpi : public abstract_clone {
public:
  clone_mpi(boost::mpi::communicator const& ctrl, boost::mpi::communicator const& work,
    boost::filesystem::path const& basedir, alps::parapack::option opt,
    clone_create_msg_t const& msg);
  virtual ~clone_mpi();

  tid_t task_id() const { return task_id_; }
  cid_t clone_id() const { return clone_id_; }

  void run();
  bool halted() const;
  clone_info const& info() const;

  void load();
  void save() const;
  void load(hdf5::archive & ar);
  void save(hdf5::archive & ar) const;

  void checkpoint();
  void suspend();

  void output() const;

protected:
  void do_checkpoint();
  void do_suspend();
  void do_halt();

  void send_info(mcmp_tag_t tag);
  void send_halted();

private:
  boost::mpi::communicator ctrl_, work_;

  tid_t task_id_;
  cid_t clone_id_;
  gid_t group_id_;

  Parameters params_;

  boost::filesystem::path basedir_;
  clone_info info_;
  std::vector<ObservableSet> measurements_;

  dump_format_t dump_format_;
  dump_policy_t dump_policy_;
  clone_timer timer_;
  clone_timer::loops_t loops_;

  boost::shared_ptr<parapack::abstract_worker> worker_;
};

#endif // ALPS_HAVE_MPI

} // end namespace alps

#endif // PARAPACK_CLONE_H
