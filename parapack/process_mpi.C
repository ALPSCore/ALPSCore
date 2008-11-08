/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2008 by Synge Todo <wistaria@comp-phys.org>
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

#include "process_mpi.h"
#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <algorithm>
#include <stdexcept>

namespace mpi = boost::mpi;

namespace alps {

//
// process_helper_mpi
//

process_helper_mpi::process_helper_mpi(mpi::communicator const& comm, int np) :
  world_d_(comm, mpi::comm_duplicate), world_u_(comm, mpi::comm_duplicate),
  np_(np), halt_stage_(0), num_halted_(0) {

  int ng = world_d_.size() / np_;

  world_d_.barrier();
  if (world_d_.size() < np_)
    boost::throw_exception(std::runtime_error("inconsistent number of processes"));

  std::vector<int> colors(world_d_.size());
  if (world_d_.size() % np_ == 0)
    for (int i = 0; i < world_d_.size(); ++i) colors[i] = i / np_;
  else
    for (int i = 0; i < world_d_.size(); ++i)
      colors[i] = (i > 0 && i <= world_d_.size() - (world_d_.size() % np_)) ?
        ((i - 1) / np_) : MPI_UNDEFINED;
  work_ = world_d_.split(colors[world_d_.rank()]);

  if (world_d_.rank() == 0) {
    status_.resize(ng, Free);
    procs_.resize(ng, alps::ProcessList());
    for (int i = 0; i < world_d_.size(); ++i)
      if (colors[i] != MPI_UNDEFINED)
        procs_[colors[i]].push_back(alps::Process(i));
    for (int g = 0; g < ng; ++g) free_.push(g);
  }
}

process_helper_mpi::~process_helper_mpi() {
  if (halt_stage_ != 3) std::cerr << "Warning: process(es) still not halted\n";
}

process_group process_helper_mpi::allocate() {
  if (free_.empty())
    boost::throw_exception(std::logic_error("no free processe groups available"));
  int g = free_.front();
  free_.pop();
  status_[g] = Active;
  return process_group(g, procs_[g]);
}

void process_helper_mpi::release(gid_t gid) {
  if (status_[gid] != Active)
    boost::throw_exception(std::logic_error("process group is not active"));
  status_[gid] = Free;
  free_.push(gid);
}

void process_helper_mpi::release(process_group const& g) {
  release(g.group_id);
}

void process_helper_mpi::halt() {
  if (world_d_.rank() == 0 && halt_stage_ == 0) {
    halt_stage_ = 1;
    check_halted();
  }
}

bool process_helper_mpi::check_halted() {
  if (world_d_.rank() == 0) {
    if (halt_stage_ == 1) {
      if (num_allocated() == 0) {
        num_halted_ = 1; // myself
        for (int p = 1; p < world_d_.size(); ++p) world_d_.send(p, mcmp_tag::scheduler_halt);
        halt_stage_ = 2;
      }
    }
    if (halt_stage_ == 2) {
      while (world_u_.iprobe(mpi::any_source, mcmp_tag::scheduler_halt)) {
        world_u_.recv(mpi::any_source, mcmp_tag::scheduler_halt);
        ++num_halted_;
      }
      if (num_halted_ == world_d_.size()) halt_stage_ = 3;
    }
  } else {
    if (halt_stage_ != 3 && world_d_.iprobe(0, mcmp_tag::scheduler_halt)) {
      world_d_.recv(0, mcmp_tag::scheduler_halt);
      world_u_.send(0, mcmp_tag::scheduler_halt);
      halt_stage_ = 3;
    }
  }
  return halt_stage_ == 3;
}

} // end namespace alps
