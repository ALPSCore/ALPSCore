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

#ifndef PARAPACK_CLONE_PROXY_H
#define PARAPACK_CLONE_PROXY_H

#include "clone.h"

namespace alps {

class clone_proxy {
public:
  clone_proxy(clone*& clone_ptr, boost::filesystem::path const& basedir, alps::parapack::option opt)
    : clone_ptr_(clone_ptr), basedir_(basedir), opt_(opt) {}
  bool is_local(Process const&) const { return true; }
  void start(tid_t tid, cid_t cid, thread_group const&, Parameters const& params,
    std::string const& base, bool is_new) {
    clone_ptr_ = new clone(basedir_, opt_, tid, cid, params, base, is_new);
  }
  clone_info const& info(Process const&) const {
    if (!clone_ptr_)
      boost::throw_exception(std::logic_error("clone_proxy::info()"));
    return clone_ptr_->info();
  }
  void checkpoint(Process const&) { if (clone_ptr_) clone_ptr_->checkpoint(); }
  void update_info(Process const&) const {}
  void suspend(Process const&) { if (clone_ptr_) clone_ptr_->suspend(); }
  void halt(Process const&) { /* if (clone_ptr_) clone_ptr_->halt(); */ }
  void destroy(Process const&) {
    if (clone_ptr_) {
      delete clone_ptr_;
      clone_ptr_ = 0;
    }
  }
private:
  clone*& clone_ptr_;
  boost::filesystem::path basedir_;
  alps::parapack::option opt_;
};

#ifdef ALPS_HAVE_MPI

class clone_proxy_mpi {
public:
  clone_proxy_mpi(clone_mpi*& clone_ptr, boost::mpi::communicator const& comm_ctrl,
    boost::mpi::communicator const& comm_work, boost::filesystem::path const& basedir,
    alps::parapack::option opt)
    : clone_ptr_(clone_ptr), comm_ctrl_(comm_ctrl), comm_work_(comm_work), basedir_(basedir),
      opt_(opt) {}
  bool is_local(Process const& proc) const { return proc == 0; }
  void start(tid_t tid, cid_t cid, process_group const& procs, Parameters const& params,
    std::string const& base, bool is_new) const {
    clone_create_msg_t msg(tid, cid, procs.group_id, params, base, is_new);
    bool worker_on_master = false;
    BOOST_FOREACH(Process p, procs.process_list) {
      if (p == 0)
        worker_on_master = true;
      else
        comm_ctrl_.send(p, mcmp_tag::clone_create, msg);
    }
    if (worker_on_master)
      clone_ptr_ =
        new clone_mpi(comm_ctrl_, comm_work_, basedir_, opt_, msg);
  }

  clone_info const& info(Process const& proc) const {
    if (proc != 0 || !clone_ptr_) {
      std::cerr << "clone_proxy_mpi::info()\n";
      boost::throw_exception(std::logic_error("clone_proxy_mpi::info()"));
    }
    return clone_ptr_->info();
  }

  void checkpoint(Process const& proc) {
    if (proc == 0) {
      if (clone_ptr_) clone_ptr_->checkpoint();
    } else {
      comm_ctrl_.send(proc, mcmp_tag::clone_checkpoint);
    }
  }

  void update_info(Process const& proc) const {
    if (proc == 0) {
      // nothing to do
    } else {
      comm_ctrl_.send(proc, mcmp_tag::clone_info);
    }
  }

  void suspend(Process const& proc) {
    if (proc == 0) {
      if (clone_ptr_) clone_ptr_->suspend();
    } else {
      comm_ctrl_.send(proc, mcmp_tag::clone_suspend);
    }
  }

  void halt(Process const& proc) {
    if (proc == 0) {
      // nothing to do
    } else {
      comm_ctrl_.send(proc, mcmp_tag::clone_halt);
    }
  }

  void destroy(Process const& proc) {
    if (proc == 0 && clone_ptr_) {
      delete clone_ptr_;
      clone_ptr_ = 0;
    }
  }

private:
  clone_mpi*& clone_ptr_;
  boost::mpi::communicator comm_ctrl_, comm_work_;
  boost::filesystem::path basedir_;
  alps::parapack::option opt_;
};

#endif // ALPS_HAVE_MPI

} // end namespace alps

#endif // PARAPACK_CLONE_PROXY_H
