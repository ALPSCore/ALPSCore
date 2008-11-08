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

#include "scheduler.h"
#include "job_p.h"
#include "queue.h"
#include "version.h"

#include <alps/osiris/comm.h>
#include <alps/copyright.h>
#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/regex.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <time.h>

namespace alps {

// compile_date
#if defined(__DATE__) && defined(__TIME__)
# define ALPS_COMPILE_DATE __DATE__ " " __TIME__
#else
# define ALPS_COMPILE_DATE "unknown"
#endif
std::string compile_date() { return ALPS_COMPILE_DATE; }

namespace parapack {

int start(int argc, char **argv) {

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

    scheduler::start(argc, argv);

#ifndef BOOST_NO_EXCEPTIONS
  }
  catch (const std::exception& excp) {
    std::cerr << excp.what() << std::endl;
    alps::comm_exit(true);
    return -1; }
  catch (...) {
    std::cerr << "Unknown exception occurred!" << std::endl;
    alps::comm_exit(true);
    return -1; }
#endif

  return 0;
}


int evaluate(int argc, char **argv) {

#ifndef BOOST_NO_EXCEPTIONS
  try {
#endif

    evaluate_option opt(argc, argv);
    if (!opt.valid) return -1;

    BOOST_FOREACH(std::string const& file_str, opt.jobfiles) {
      boost::filesystem::path file = complete(boost::filesystem::path(file_str)).normalize();
      boost::filesystem::path basedir = file.branch_path();
      std::string file_in_str;
      std::string file_out_str;
      std::vector<task> tasks;

      std::clog << scheduler::log_header() << "starting evaluation on "
                << alps::hostname() << std::endl;
      int t = scheduler::load_filename(file, file_in_str, file_out_str);
      if (t == 1) {
        // process all tasks
        boost::filesystem::path file_in = complete(boost::filesystem::path(file_in_str), basedir);
        boost::filesystem::path file_out = complete(boost::filesystem::path(file_out_str), basedir);
        std::string simname;
        scheduler::load_tasks(file_in, file_out, basedir, /* check_parameter = */ false,
                              simname, tasks);
        std::clog << "  master input file  = " << file_in.native_file_string() << std::endl
                  << "  master output file = " << file_out.native_file_string() << std::endl;
        scheduler::print_taskinfo(std::clog, tasks);
        BOOST_FOREACH(task& t, tasks) t.evaluate();
      } else {
        // process one task
        task t(file);
        t.evaluate();
      }
      std::clog << scheduler::log_header() << "all tasks evaluated\n";
    }

#ifndef BOOST_NO_EXCEPTIONS
  }
  catch (const std::exception& excp) {
    std::cerr << excp.what() << std::endl;
    return -1; }
  catch (...) {
    std::cerr << "known exception occurred!" << std::endl;
    return -1; }
#endif

  return 0;
}

namespace scheduler {

void print_copyright(std::ostream& os) {
  worker_factory::print_copyright(os);
  os << std::endl << "using " << PARAPACK_COPYRIGHT << std::endl;
  alps::print_copyright(os);
}

void print_license(std::ostream& os) {
  os << "Please look at the file LICENSE for the license conditions.\n";
}

std::string alps_version() {
  return ALPS_VERSION_STRING "; " PARAPACK_VERSION_STRING "; configured on " ALPS_CONFIG_HOST " by " ALPS_CONFIG_USER "; compiled on " ALPS_COMPILE_DATE;
}

std::string log_header() {
  return
    std::string("[") + to_simple_string(boost::posix_time::second_clock::local_time()) + "]: ";
}

std::string clone_name(alps::tid_t tid, alps::cid_t cid) {
  return std::string("clone[") + boost::lexical_cast<std::string>(tid+1) + ',' +
    boost::lexical_cast<std::string>(cid+1) + ']';
}

std::string pg_name(alps::gid_t gid) {
  return std::string("processgroup[") + boost::lexical_cast<std::string>(gid+1) + ']';
}

void print_taskinfo(std::ostream& os, std::vector<alps::task> const& tasks) {
  uint32_t num_new = 0;
  uint32_t num_running = 0;
  uint32_t num_continuing = 0;
  uint32_t num_suspended = 0;
  uint32_t num_finished = 0;
  uint32_t num_completed = 0;
  BOOST_FOREACH(alps::task const& t, tasks) {
    switch (t.status()) {
    case alps::task_status::NotStarted :
      ++num_new;
      break;
    case alps::task_status::Running :
      ++num_running;
      break;
    case alps::task_status::Continuing :
      ++num_continuing;
      break;
    case alps::task_status::Suspended :
      ++num_suspended;
      break;
    case alps::task_status::Finished :
      ++num_finished;
      break;
    case alps::task_status::Completed :
      ++num_completed;
      break;
    default :
      break;
    }
  }
  os << log_header() << "task status: "
     << "total number of tasks = " << tasks.size() << std::endl
     << "  new = " << num_new
     << ", running = " << num_running
     << ", continuing = " << num_continuing
     << ", suspended = " << num_suspended
     << ", finished = " << num_finished
     << ", completed = " << num_completed << std::endl;
}

int load_filename(boost::filesystem::path const& file, std::string& file_in_str,
  std::string& file_out_str) {
  bool is_master;
  alps::filename_xml_handler handler(file_in_str, file_out_str, is_master);
  alps::XMLParser parser(handler);
  parser.parse(file);
  if (is_master && file_in_str.empty())
    file_in_str = regex_replace(file_out_str, boost::regex("\\.out\\.xml$"), ".in.xml");
  return is_master ? 1 : 2;
}

void load_version(boost::filesystem::path const& file,
  std::vector<std::pair<std::string, std::string> >& versions) {
  alps::version_xml_handler handler(versions);
  alps::XMLParser parser(handler);
  parser.parse(file);
}

void load_tasks(boost::filesystem::path const& file_in,
  boost::filesystem::path const& file_out, boost::filesystem::path const& basedir,
  bool check_parameter, std::string& simname, std::vector<alps::task>& tasks) {
  tasks.clear();
  alps::job_tasks_xml_handler handler(simname, tasks, basedir);
  alps::XMLParser parser(handler);
  if (check_parameter || !exists(file_out))
    parser.parse(file_in);
  else
    parser.parse(file_out);

  if (check_parameter && exists(file_out)) {
    std::vector<alps::task> tasks_out;
    alps::job_tasks_xml_handler handler_out(simname, tasks_out, basedir);
    alps::XMLParser parser_out(handler_out);
    parser_out.parse(file_out);

    if (tasks.size() == tasks_out.size()) {
    } else if (tasks.size() > tasks_out.size()) {
      std::clog << "Info: number of parameter sets has been increased from " << tasks_out.size()
                << " to " << tasks.size() << std::endl;
    } else {
      std::clog << "Warning: number of parameter sets has been decreased from " << tasks_out.size()
                << " to " << tasks.size() << std::endl;
    }

    int nc = std::min(tasks.size(), tasks_out.size());
    for (int i = 0; i < nc; ++i) {
      if (tasks[i].file_in_str() != tasks_out[i].file_in_str()) {
        std::cerr << "Error: input XML filename of task[" << i << "] has been modified\n";
        boost::throw_exception(std::runtime_error("check parameter"));
      }
      if (tasks[i].file_out_str() != tasks_out[i].file_out_str()) {
        std::cerr << "Error: output XML filename of task[" << i << "] has been modified\n";
        boost::throw_exception(std::runtime_error("check parameter"));
      }
    }
    BOOST_FOREACH(alps::task& t, tasks) t.check_parameter();
  }
}

void save_tasks(boost::filesystem::path const& file, std::string const& simname,
  std::string const& file_in_str, std::string const& file_out_str,
  std::vector<alps::task>& tasks) {
  alps::job_xml_writer(file, simname, file_in_str, file_out_str, alps_version(),
    worker_factory::version(), tasks, true);
}

} // end namespace scheduler
} // end namespace parapack
} // end namespace alps
