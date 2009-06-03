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

#ifndef PARAPACK_PROCESS_MPI_H
#define PARAPACK_PROCESS_MPI_H

#include "process.h"
#include <queue>
#include <set>
#include <utility>
#include <vector>
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>

namespace mpi = boost::mpi;

namespace alps {

//
// process_helper_mpi
//

class process_helper_mpi {
private:
  enum group_state { Free, Active };

public:
  process_helper_mpi(mpi::communicator const& comm, int np);
  ~process_helper_mpi();

  mpi::communicator const& comm_ctrl() const { return ctrl_; }
  mpi::communicator const& comm_work() const { return work_; }
  int num_procs_per_group() const { return np_; }

  int num_total_processes() const { return ctrl_.size(); }
  int num_groups() const { return status_.size(); }
  int num_free() const { return free_.size(); }
  int num_allocated() const { return num_groups() - num_free(); }

  process_group allocate();
  void release(gid_t gid);
  void release(process_group const& g);

  void halt();
  bool is_halting() { return halt_stage_ != 0; }
  bool check_halted();

private:
  mpi::communicator ctrl_, work_;

  // number of processes in each group
  int np_;

  // process ID in each group
  std::vector<alps::ProcessList> procs_;

  // status of each group
  std::vector<group_state> status_;

  // list of free gruops
  std::queue<int> free_;

  // 0: normal
  // 1: waiting for allocated group
  // 2: waiting for halting procs
  // 3: halted
  int halt_stage_;

  // work array for halting
  int num_halted_;
};

//
// collective communication
//

template<typename T>
void collect_vector(mpi::communicator const& comm, std::vector<int> const& nelms,
  std::vector<int> const& offsets, std::vector<T> const& source, std::vector<T>& target) {
  if (comm.rank() == 0) {
    target.resize(std::accumulate(nelms.begin(), nelms.end(), 0));
    std::copy(source.begin(), source.begin() + nelms[0], target.begin());
    for (int p = 1; p < nelms.size(); ++p)
      comm.recv(p, 0, &target[offsets[p]], nelms[p]);
  } else {
    comm.send(0, 0, &source[0], nelms[comm.rank()]);
  }
}

template<typename T>
void distribute_vector(mpi::communicator const& comm, std::vector<int> const& nelms,
  std::vector<int> const& offsets, std::vector<T> const& source, std::vector<T>& target) {
  if (comm.rank() == 0) {
    for (int p = 1; p < nelms.size(); ++p)
      comm.send(p, 0, &source[offsets[p]], nelms[p]);
    target.resize(nelms[0]);
    std::copy(source.begin(), source.begin() + nelms[0], target.begin());
  } else {
    target.resize(nelms[comm.rank()]);
    comm.recv(0, 0, &target[0], nelms[comm.rank()]);
  }
}

} // end namespace alps

#endif // PARAPACK_PROCESS_MPI_H
