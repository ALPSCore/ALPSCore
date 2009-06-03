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

#ifndef PARAPACK_CLONE_MPI_H
#define PARAPACK_CLONE_MPI_H

#include "clone.h"
#include "process_mpi.h"

namespace alps {

struct clone_create_msg_t {
  clone_create_msg_t() {}
  clone_create_msg_t(tid_t tid, cid_t cid, gid_t gid, Parameters const& p, std::string const& bsd,
                     std::string const& bs, clone_timer::duration_t const& it, bool in)
    : task_id(tid), clone_id(cid), group_id(gid), params(p), basedir(bsd), base(bs),
      check_interval(it), is_new(in) {}
  tid_t task_id;
  cid_t clone_id;
  gid_t group_id;
  Parameters params;
  std::string basedir;
  std::string base;
  clone_timer::duration_t check_interval;
  bool is_new;
  BOOST_SERIALIZATION_SPLIT_MEMBER()
  template<class Archive>
  void save(Archive & ar, const unsigned int) const {
    long sec = check_interval.total_seconds();
    ar & task_id & clone_id & group_id & params & basedir & base & sec & is_new;
  }
  template<class Archive>
  void load(Archive & ar, const unsigned int) {
    long sec;
    ar & task_id & clone_id & group_id & params & basedir & base & sec & is_new;
    check_interval = boost::posix_time::seconds(sec);
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
    clone_create_msg_t const& msg);
  virtual ~clone_mpi();

  tid_t task_id() const { return task_id_; }
  cid_t clone_id() const { return clone_id_; }

  void run();
  bool halted() const;
  clone_info const& info() const;

  void load(IDump& dump);
  void save(ODump& dump) const;

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

  clone_timer timer_;
  clone_timer::loops_t loops_;

  boost::shared_ptr<parapack::abstract_worker> worker_;
};

} // end namespace alps

#endif // PARAPACK_CLONE_MPI_H
