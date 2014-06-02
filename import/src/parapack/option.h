/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
