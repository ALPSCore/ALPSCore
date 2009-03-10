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

#ifndef PARAPACK_CLONE_PROXY_MPI_H
#define PARAPACK_CLONE_PROXY_MPI_H

#include "process_mpi.h"
#include "clone_mpi.h"

namespace alps {

class clone_proxy_mpi {
public:
  clone_proxy_mpi(boost::mpi::communicator const& comm) : comm_(comm) {}

  void start(tid_t tid, cid_t cid, process_group const& procs, Parameters const& params,
    boost::filesystem::path const& basedir, std::string const& base, bool is_new) const {
    clone_create_msg_t msg(tid, cid, procs.group_id, params, basedir.native_file_string(), base,
                           is_new);
    BOOST_FOREACH(Process p, procs.process_list) comm_.send(p, mcmp_tag::clone_create, msg);
  }

  void checkpoint(Process const& proc) const {
    comm_.send(proc, mcmp_tag::clone_checkpoint);
  }

  void update_info(Process const& proc) const {
    comm_.send(proc, mcmp_tag::clone_info);
  }

  void suspend(Process const& proc) const {
    comm_.send(proc, mcmp_tag::clone_suspend);
  }

  void halt(Process const& proc) const {
    comm_.send(proc, mcmp_tag::clone_halt);
  }

private:
  boost::mpi::communicator comm_;
};

} // end namespace alps

#endif // PARAPACK_CLONE_PROXY_MPI_H
