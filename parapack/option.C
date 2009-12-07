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

#include "option.h"
#include "process.h"

#include <boost/filesystem/operations.hpp>
#include <boost/foreach.hpp>
#include <boost/program_options.hpp>

#ifdef _OPENMP
# include <omp.h>
#endif

namespace alps {
namespace parapack {

namespace po = boost::program_options;
namespace pt = boost::posix_time;

option::option(int argc, char** argv, int np, int pid)
  : time_limit(), num_total_threads(np == 1 ? max_threads() : np), threads_per_clone(1),
    check_parameter(false), auto_evaluate(false), evaluate_only(false), jobfiles(), valid(true) {

  check_interval = pt::millisec(100);
  checkpoint_interval = pt::seconds(3600);
  report_interval = pt::seconds(600);

  po::options_description desc("Allowed options");
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
    if (pid == 0) std::cerr << "Error: unknown command line option(s)\n";
    valid = false;
    return;
  }

  if (vm.count("auto-evaluate"))
    auto_evaluate = true;
  if (vm.count("check-parameter"))
    check_parameter = true;
  if (vm.count("check-interval"))
    check_interval = pt::millisec(vm["check-interval"].as<int>());
  if (vm.count("checkpoint-interval"))
    checkpoint_interval = pt::seconds(vm["checkpoint-interval"].as<int>());
  if (vm.count("report-interval"))
    report_interval = pt::seconds(vm["report-interval"].as<int>());
  if (vm.count("evaluate"))
    evaluate_only = true;
  if (vm.count("time-limit"))
    time_limit = pt::seconds(vm["time-limit"].as<int>());
  if (vm.count("threads-per-clone"))
    threads_per_clone = vm["threads-per-clone"].as<int>();
  if (vm.count("total-threads")) {
    if (vm["total-threads"].as<std::string>() == "auto") {
      num_total_threads = np * max_threads();
    } else {
      num_total_threads = boost::lexical_cast<int>(vm["total-threads"].as<std::string>());
    }
  }
  if (pid == 0) {
    if (vm.count("help")) {
      std::cout << desc << std::endl;
      valid = false;
    }
    if (vm.count("license")) {
      std::cout << "license" << std::endl;
      valid = false;
    }
    if (num_total_threads < 1 || threads_per_clone < 1 ||
        threads_per_clone > num_total_threads || num_total_threads % threads_per_clone != 0) {
      std::cerr << "Error: invalid number of threads\n";
      valid = false;
    }
  }
  if (vm.count("input-file"))
    jobfiles = vm["input-file"].as<std::vector<std::string> >();
  if (pid == 0) {
    BOOST_FOREACH(std::string const& file, jobfiles) {
      if (!exists(complete(boost::filesystem::path(file)))) {
        std::cerr << "Error: file not found: " << file << std::endl;
        valid = false;
      }
    }
  }
}

evaluate_option::evaluate_option(int argc, char** argv) : jobfiles(), valid(true) {
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help,h", "produce help message")
    ("license,l", "print license conditions")
    ("input-file", po::value<std::vector<std::string> >(), "input master XML files");
  po::positional_options_description p;
  p.add("input-file", -1);

  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << desc << std::endl;
    valid = false;
  }
  if (vm.count("license")) {
    std::cout << "license" << std::endl;
    valid = false;
  }

  if (vm.count("input-file"))
    jobfiles = vm["input-file"].as<std::vector<std::string> >();
  BOOST_FOREACH(std::string const& file, jobfiles) {
    if (!exists(complete(boost::filesystem::path(file)))) {
      std::cerr << "Error: file not found: " << file << std::endl;
      valid = false;
    }
  }
}

} // end namespace parapack
} // end namespace alps
