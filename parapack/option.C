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

#include "option.h"

namespace alps {
namespace parapack {

namespace po = boost::program_options;
namespace pt = boost::posix_time;

option::option(int argc, char** argv)
  : desc("Allowed options"), has_time_limit(false), time_limit(), check_interval(pt::millisec(100)),
    checkpoint_interval(pt::seconds(3600)), report_interval(pt::seconds(600)),
    default_total_threads(true), auto_total_threads(false), num_total_threads(1),
    threads_per_clone(1), auto_evaluate(false), evaluate_only(false),
    use_mpi(false), jobfiles(), valid(true), show_help(false), show_license(false) {
  desc.add_options()
    ("help,h", "produce help message")
    ("license,l", "print license conditions")
    ("auto-evaluate", "evaluate observables upon halting")
    ("check-parameter", "perform parameter checking")
    ("check-interval", po::value<int>(),
     "time between internal status check [unit = millisec; default = 100ms]")
    ("checkpoint-interval", po::value<int>(),
     "time between checkpointing [unit = sec; default = 3600s]")
    ("evaluate", "evaluation mode")
    ("mpi", "run in parallel using MPI")
    ("Nmin", "obsolete")
    ("Nmax", "obsolete")
    ("report-interval", po::value<int>(),
     "time between progress report of clones [unit = sec; default = 600s]")
    ("time-limit,T", po::value<int>(),
     "time limit for the simulation [unit = sec; defulat = no time limit]")
    ("Tmin", "obsolete")
    ("Tmax", "obsolete")
    ("threads-per-clone,p", po::value<int>(),
     "number of threads for each clone [default = 1]")
    ("total-threads,r", po::value<std::string>(),
     "total number of threads [integer or 'auto'; default = total number of processes]")
    ("input-file", po::value<std::vector<std::string> >(),
     "input master XML files");
  po::positional_options_description p;
  p.add("input-file", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
  }
  catch (...) {
    valid = false;
    return;
  }

  if (vm.count("help"))
    show_help = true;
  if (vm.count("license"))
    show_license = true;
  if (vm.count("auto-evaluate"))
    auto_evaluate = true;
  if (vm.count("check-interval"))
    check_interval = pt::millisec(vm["check-interval"].as<int>());
  if (vm.count("checkpoint-interval"))
    checkpoint_interval = pt::seconds(vm["checkpoint-interval"].as<int>());
  if (vm.count("report-interval"))
    report_interval = pt::seconds(vm["report-interval"].as<int>());
  if (vm.count("mpi"))
    use_mpi = true;
  if (vm.count("evaluate"))
    evaluate_only = true;
  if (vm.count("time-limit")) {
    has_time_limit = true;
    time_limit = pt::seconds(vm["time-limit"].as<int>());
  }
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
  if (vm.count("input-file"))
    jobfiles = vm["input-file"].as<std::vector<std::string> >();
}

void option::print(std::ostream& os) const { desc.print(os); }

evaluate_option::evaluate_option(int argc, char** argv)
  : desc("Allowed options"), jobfiles(), valid(true), show_help(false), show_license(false) {
  desc.add_options()
    ("help,h", "produce help message")
    ("license,l", "print license conditions")
    ("input-file", po::value<std::vector<std::string> >(), "input master XML files");
  po::positional_options_description p;
  p.add("input-file", -1);

  po::variables_map vm;
  try {
    po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
    po::notify(vm);
  }
  catch (...) {
    valid = false;
    return;
  }

  if (vm.count("help"))
    show_help = true;
  if (vm.count("license"))
    show_license = true;
  if (vm.count("input-file"))
    jobfiles = vm["input-file"].as<std::vector<std::string> >();
}

void evaluate_option::print(std::ostream& os) const { desc.print(os); }

} // end namespace parapack
} // end namespace alps
