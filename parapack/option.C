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

#include "option.h"

namespace alps {
namespace parapack {

namespace po = boost::program_options;
namespace pt = boost::posix_time;

option::option(int argc, char** argv, bool eval)
  : desc("Allowed options"), for_evaluate(eval), show_help(false),
    show_license(false), time_limit(pt::pos_infin),
    check_interval(pt::millisec(100)), checkpoint_interval(pt::seconds(3600)),
    report_interval(pt::seconds(600)), vmusage_interval(pt::pos_infin),
    use_termfile(false), auto_evaluate(true), evaluate_only(false),
    dump_format(dump_format::hdf5), dump_policy(dump_policy::RunningOnly),
    task_range(), write_xml(false),
    use_mpi(false), default_total_threads(true), auto_total_threads(false),
    num_total_threads(1), threads_per_clone(1), jobfiles(), valid(true) {
  desc.add_options()
    ("help,h", "produce help message")
    ("license,l", "print license conditions")
    ("dump-format", po::value<std::string>(),
     "format for dumping info, parameter, and measurements [hdf5 (default), xdr]")
    ("task-range", po::value<std::string>(),
     "specify range of task indices to be processed, e.g. [2:5]")
    ("write-xml", "write results to XML files")
    ("input-file", po::value<std::vector<std::string> >(),
     "input master XML files");
  if (!for_evaluate) {
    desc.add_options()
      ("auto-evaluate", "evaluate observables upon halting [default = true]")
      ("check-parameter", "perform parameter checking")
      ("check-interval", po::value<int>(),
       "time between internal status check [unit = millisec; default = 100ms]")
      ("checkpoint-interval", po::value<int>(),
       "time between checkpointing [unit = sec; default = 3600s]")
      ("dump-policy", po::value<std::string>(),
       "policy for dumping user checkpoint data [running (default), never, all]")
      ("enable-termination-file", "enable termination file support (*.term)")
      ("evaluate", "evaluation mode")
      ("mpi", "run in parallel using MPI")
      ("Nmin", po::value<int>(), "obsolete")
      ("Nmax", po::value<int>(), "obsolete")
      ("no-evaluate", "prevent evaluating observables upon halting")
      ("report-interval", po::value<int>(),
       "time between progress report of clones [unit = sec; default = 600s]")
      ("vmusage-interval", po::value<int>(),
       "time between virtual memory usage report of processes [unit = sec; default = none]")
      ("time-limit,T", po::value<int>(),
       "time limit for the simulation [unit = sec; default = no time limit]")
      ("Tmin", po::value<int>(), "obsolete")
      ("Tmax", po::value<int>(), "obsolete")
      ("threads-per-clone,p", po::value<int>(),
       "number of threads for each clone [default = 1]")
      ("total-threads,r", po::value<std::string>(),
       "total number of threads [integer or 'auto'; default = total number of processes]");
  }
  po::positional_options_description p;
  p.add("input-file", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
  }
  catch (std::exception& e) {
    valid = false;
    std::cerr << e.what() << std::endl;
    return;
  }

  if (vm.count("help"))
    show_help = true;
  if (vm.count("license"))
    show_license = true;
  if (vm.count("dump-format")) {
    std::string value = vm["dump-format"].as<std::string>();
    if (value == "hdf5")
      dump_format = dump_format::hdf5;
    else if (value == "xdr")
      dump_format = dump_format::xdr;
    else {
      valid = false;
      return;
    }
  }
  if (vm.count("task-range"))
    task_range = task_range_t(vm["task-range"].as<std::string>());
  if (vm.count("write-xml"))
    write_xml = true;
  if (vm.count("input-file"))
    jobfiles = vm["input-file"].as<std::vector<std::string> >();
  if (!for_evaluate) {
    if (vm.count("auto-evaluate"))
      auto_evaluate = true;
    if (vm.count("no-evaluate"))
      auto_evaluate = false;
    if (vm.count("check-interval"))
      check_interval = pt::millisec(vm["check-interval"].as<int>());
    if (vm.count("checkpoint-interval"))
      checkpoint_interval = pt::seconds(vm["checkpoint-interval"].as<int>());
    if (vm.count("enable-termination-file"))
      use_termfile = true;
    if (vm.count("dump-policy")) {
      std::string value = vm["dump-policy"].as<std::string>();
      if (value == "never")
        dump_policy = dump_policy::Never;
      else if (value == "running")
        dump_policy = dump_policy::RunningOnly;
      else if (value == "all")
        dump_policy = dump_policy::All;
      else {
        valid = false;
        return;
      }
    }
    if (vm.count("report-interval"))
      report_interval = pt::seconds(vm["report-interval"].as<int>());
    if (vm.count("vmusage-interval"))
      vmusage_interval = pt::seconds(vm["vmusage-interval"].as<int>());
    if (vm.count("mpi"))
      use_mpi = true;
    if (vm.count("evaluate"))
      evaluate_only = true;
    if (vm.count("time-limit"))
      time_limit = pt::seconds(vm["time-limit"].as<int>());
    if (vm.count("threads-per-clone"))
      threads_per_clone = vm["threads-per-clone"].as<int>();
    if (vm.count("total-threads")) {
      default_total_threads = false;
      if (vm["total-threads"].as<std::string>() == "auto") {
        auto_total_threads = true;
      } else {
        num_total_threads = boost::lexical_cast<int>(vm["total-threads"].as<std::string>());
      }
    }
  }
}

void option::print(std::ostream& os) const { desc.print(os); }

void option::print_summary(std::ostream& os, std::string const& prefix) const {
  if (!for_evaluate) {
    os << prefix << "auto evaluation = " << (auto_evaluate ? "yes" : "no") << std::endl;
    os << prefix << "time limit = ";
    if (!time_limit.is_special())
      os << time_limit.total_seconds() << " seconds\n";
    else
      os << "unlimited\n";
    os << prefix << "interval between checkpointing  = "
       << checkpoint_interval.total_seconds() << " seconds\n";
    os << prefix << "interval between progress report = "
       << report_interval.total_seconds() << " seconds\n";
    os << prefix << "interval between vmusage report = ";
    if (!vmusage_interval.is_special())
      os << vmusage_interval.total_seconds() << " seconds\n";
    else
      os << "infinity\n";
  }
  os << prefix << "task range = ";
  if (task_range.valid())
    os << task_range << std::endl;
  else
    os << "all\n";
  os << prefix << "worker dump format = " << dump_format::to_string(dump_format) << std::endl;
  if (!for_evaluate) {
    os << prefix << "worker dump policy = " << dump_policy::to_string(dump_policy) << std::endl;
  }
}

} // end namespace parapack
} // end namespace alps
