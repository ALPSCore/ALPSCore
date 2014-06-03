/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_SCHEDULER_H
#define PARAPACK_SCHEDULER_H

#include <alps/config.h>
#include "worker_factory.h"
#include "option.h"
#include "job.h"
#include <iostream>

namespace alps {

/// return the compile date of ALPS/parapack
ALPS_DECL std::string compile_date();

namespace parapack {

ALPS_DECL int start(int argc, char **argv);

ALPS_DECL int evaluate(int argc, char **argv);

ALPS_DECL int run_sequential(int argc, char **argv);

ALPS_DECL int run_sequential_mpi(int argc, char **argv);

ALPS_DECL int start_sgl(int argc, char **argv);

ALPS_DECL int start_mpi(int argc, char **argv);

ALPS_DECL void print_copyright(std::ostream& os = std::cout);

ALPS_DECL void print_license(std::ostream& os = std::cout);

ALPS_DECL std::string alps_version();

ALPS_DECL void print_taskinfo(std::ostream& os, std::vector<alps::task> const& tasks,
  task_range_t const& task_range);

// return 1 for job XML (<JOB>) file or 2 for task XML (<SIMULATION>)
ALPS_DECL int load_filename(boost::filesystem::path const& file, std::string& file_in_str,
  std::string& file_out_str);

ALPS_DECL void load_version(boost::filesystem::path const& file,
  std::vector<std::pair<std::string, std::string> >& versions);

ALPS_DECL void load_tasks(boost::filesystem::path const& file_in,
  boost::filesystem::path const& file_out, boost::filesystem::path const& basedir,
  std::string& simname, std::vector<alps::task>& tasks, bool check_parameter,
  alps::parapack::option const& opt);

ALPS_DECL void load_tasks(boost::filesystem::path const& file_in,
  boost::filesystem::path const& file_out, boost::filesystem::path const& basedir,
  std::string& simname, std::vector<alps::task>& tasks);

ALPS_DECL void save_tasks(boost::filesystem::path const& file, std::string const& simname,
  std::string const& file_in_str, std::string const& file_out_str, std::vector<alps::task>& tasks);

} // namespace parapack
} // namespace alps

#endif // PARAPACK_SCHEDULER_H
