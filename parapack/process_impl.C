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

#include "process.h"
#ifdef ALPS_HAVE_MPI
#include "staging.h"
# include <boost/lexical_cast.hpp>
# include <boost/throw_exception.hpp>
# include <algorithm>
# include <stdexcept>
#endif // ALPS_HAVE_MPI

#ifdef ALPS_HAVE_MPI

namespace alps {

//
// process_helper_mpi
//

process_helper_mpi::process_helper_mpi(boost::mpi::communicator const& comm, int np) :
  ctrl_(comm, boost::mpi::comm_duplicate), np_(np), halt_stage_(0), num_halted_(0) {

  int ng = comm.size() / np_;

  comm.barrier();
  if (comm.size() < np_) {
    std::cerr << "inconsistent number of processes\n";
    boost::throw_exception(std::runtime_error("inconsistent number of processes"));
  }

  std::vector<int> colors(comm.size());
  if (comm.size() % np_ == 0)
    for (int i = 0; i < comm.size(); ++i) colors[i] = i / np_;
  else
    for (int i = 0; i < comm.size(); ++i)
      colors[i] = (i > 0 && i <= comm.size() - (comm.size() % np_)) ?
        ((i - 1) / np_) : MPI_UNDEFINED;
  work_ = comm.split(colors[comm.rank()]);

  if (comm.rank() == 0) {
    status_.resize(ng, Free);
    procs_.resize(ng, alps::ProcessList());
    for (int i = 0; i < comm.size(); ++i)
      if (colors[i] != MPI_UNDEFINED)
        procs_[colors[i]].push_back(alps::Process(i));
    for (int g = 0; g < ng; ++g) free_.push_back(g);
  }

  head_ = comm.split((work_ && work_.rank() == 0) ? 1 : MPI_UNDEFINED);
}

process_helper_mpi::~process_helper_mpi() {
  if (halt_stage_ != 3) std::cerr << "Warning: process(es) still not halted\n";
}

process_group process_helper_mpi::allocate() {
  if (free_.empty()) {
    std::cerr << "no free processe groups available\n";
    boost::throw_exception(std::logic_error("no free processe groups available"));
  }
  int g = free_.front();
  free_.pop_front();
  status_[g] = Active;
  return process_group(g, procs_[g]);
}

process_group process_helper_mpi::allocate(parapack::suspended_queue_t& sq) {
  if (free_.empty()) {
    std::cerr << "no free processe groups available\n";
    boost::throw_exception(std::logic_error("no free processe groups available"));
  }
 
  std::list<int>::iterator ig = std::find(free_.begin(), free_.end(), sq.get<2>());
  int g = *ig;
  free_.erase(ig);
  status_[g] = Active;
    
  return process_group(g, procs_[g]);
}

void process_helper_mpi::release(gid_t gid) {
  if (status_[gid] != Active) {
    std::cerr << "process group is not active\n";
    boost::throw_exception(std::logic_error("process group is not active"));
  }
  status_[gid] = Free;
  free_.push_back(gid);
}

void process_helper_mpi::release(process_group const& g) {
  release(g.group_id);
}

void process_helper_mpi::halt() {
  if (ctrl_.rank() == 0 && halt_stage_ == 0) {
    halt_stage_ = 1;
    check_halted();
  }
}

bool process_helper_mpi::check_halted() {
  if (ctrl_.rank() == 0) {
    if (halt_stage_ == 1) {
      if (num_allocated() == 0) {
        num_halted_ = 1; // myself
        for (int p = 1; p < ctrl_.size(); ++p) ctrl_.send(p, mcmp_tag::scheduler_halt);
        halt_stage_ = 2;
      }
    }
    if (halt_stage_ == 2) {
      while (ctrl_.iprobe(boost::mpi::any_source, mcmp_tag::scheduler_halt)) {
        ctrl_.recv(boost::mpi::any_source, mcmp_tag::scheduler_halt);
        ++num_halted_;
      }
      if (num_halted_ == ctrl_.size()) halt_stage_ = 3;
    }
  } else {
    if (halt_stage_ != 3 && ctrl_.iprobe(0, mcmp_tag::scheduler_halt)) {
      ctrl_.recv(0, mcmp_tag::scheduler_halt);
      ctrl_.send(0, mcmp_tag::scheduler_halt);
      halt_stage_ = 3;
    }
  }
  return halt_stage_ == 3;
}

} // end namespace alps

#endif // ALPS_HAVE_MPI
