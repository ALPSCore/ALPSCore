/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2013 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_OPTION_H
#define PARAPACK_OPTION_H

#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/program_options.hpp>
#include "types.h"

namespace alps {
namespace parapack {

struct option {
  option(int argc, char** argv, bool for_evaluate = false);
  boost::program_options::options_description desc;
  bool for_evaluate;
  bool show_help, show_license;
  boost::posix_time::time_duration time_limit;
  boost::posix_time::time_duration check_interval, checkpoint_interval, report_interval;
  boost::posix_time::time_duration vmusage_interval;
  bool use_termfile;
  bool auto_evaluate, evaluate_only;
  dump_format_t dump_format;
  dump_policy_t dump_policy;
  task_range_t task_range;
  bool write_xml;
  bool use_mpi, default_total_threads, auto_total_threads;
  int num_total_threads, threads_per_clone;
  std::vector<std::string> jobfiles;
  bool valid;
  void print(std::ostream& os) const;
  void print_summary(std::ostream& os, std::string const& prefix = "") const;
};

} // end namespace parapack
} // end namespace alps

#endif // PARAPACK_OPTION_H
