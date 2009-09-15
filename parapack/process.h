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

#ifndef PARAPACK_PROCESS_H
#define PARAPACK_PROCESS_H

#include "types.h"
#include <alps/osiris.h>

#ifdef _OPENMP
# include <omp.h>
#endif

namespace alps {

struct process_group {
  process_group() : group_id(0), process_list() {}
  process_group(gid_t gid) : group_id(gid), process_list() {}
  process_group(gid_t gid, ProcessList const& plist) : group_id(gid), process_list(plist) {}
  Process master() const { return process_list[0]; }
  gid_t group_id;
  ProcessList process_list;
};

class process_helper {
public:
  process_helper() : halted_(false) {}
  void halt() { halted_ = true; }
  bool is_halting() const { return halted_; }
  bool check_halted() const { return halted_; }
private:
  bool halted_;
};

struct thread_group {
  thread_group() : group_id(0) {}
  thread_group(gid_t gid) : group_id(gid) {}
  Process master() const { return Process(); }
  gid_t group_id;
};

inline int thread_id() {
#ifndef _OPENMP
  return 0;
#else
  return omp_get_thread_num();
#endif
}

} // end namespace alps

#endif // PARAPACK_PROCESS_H
