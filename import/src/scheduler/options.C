/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

/* $Id$ */

#include <alps/scheduler/options.h>
#include <alps/utility/copyright.hpp>
#include <boost/limits.hpp>
#include <boost/throw_exception.hpp>
#include <boost/program_options.hpp>
#include <stdexcept>

namespace alps {
namespace scheduler {

namespace po = boost::program_options;

NoJobfileOptions::NoJobfileOptions()
  : min_check_time(60),
    max_check_time(900),
    checkpoint_time(1800),
    min_cpus(1),
    max_cpus(1),
    time_limit(0.),
    use_mpi(false),
    valid(true), // shall we really run?
    write_xml(false)
{
}

NoJobfileOptions::NoJobfileOptions(int argc, char** argv) 
  : programname(std::string(argv[0])),
    use_mpi(false),
    valid(true), // shall we really run?
    write_xml(false)
{
  if (argc) {
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("license,l", "print license conditions") 
    ("mpi", "run in parallel using MPI") 
    ("checkpoint-time", po::value<double>(&checkpoint_time)->default_value(1800),"time between checkpoints")
    ("Tmin", po::value<double>(&min_check_time)->default_value(60),"minimum time between checks whether a simulation is finished")
    ("Tmax", po::value<double>(&max_check_time)->default_value(900),"maximum time between checks whether a simulation is finished")
    ("time-limit,T", po::value<double>(&time_limit)->default_value(0),"time limit for the simulation")
    ("Nmin", po::value<int>(&min_cpus)->default_value(1),"minimum number of CPUs per simulation")
    ("Nmax", po::value<int>(&max_cpus)->default_value(std::numeric_limits<int>::max()),"maximum number of CPUs per simulation")
    ("write-xml","write results to XML files");
  po::positional_options_description p;
  p.add("input-file", 1);
  
  bool error=false;
  po::variables_map vm;
  try {
	  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
	  po::notify(vm);  
  }
  catch (...) {
  error = true;
  }

  if (error || vm.count("help")) {
    std::cout << desc << "\n";
    valid=false;
  }
  if (vm.count("license")) {
    print_license(std::cout);
    valid=false;
  }
  if (!valid)
    return;

  if (vm.count("mpi")) {
    use_mpi = true;
  }

#ifdef ALPS_HAVE_HDF5
  if (vm.count("write-xml"))
    write_xml = true;
#else
  write_xml=true;
#endif

  if(min_cpus>max_cpus)
    boost::throw_exception(std::runtime_error("Minimum number of CPUs larger than maximum number of CPU"));
  if(min_check_time>max_check_time)
    boost::throw_exception(std::runtime_error("Minimum time between checks larger than maximum time"));
  }
}

Options::Options()
  : NoJobfileOptions()
{
}

Options::Options(int argc, char** argv) 
{
  programname = std::string(argv[0]);
  valid = true;
  if (argc) {
  std::string filename;
  
  po::options_description desc("Allowed options");
  desc.add_options()
    ("help", "produce help message")
    ("license,l", "print license conditions") 
    ("mpi", "run in parallel using MPI") 
    ("checkpoint-time", po::value<double>(&checkpoint_time)->default_value(1800),"time between checkpoints")
    ("Tmin", po::value<double>(&min_check_time)->default_value(60),"minimum time between checks whether a simulation is finished")
    ("Tmax", po::value<double>(&max_check_time)->default_value(900),"maximum time between checks whether a simulation is finished")
    ("time-limit,T", po::value<double>(&time_limit)->default_value(0),"time limit for the simulation")
    ("Nmin", po::value<int>(&min_cpus)->default_value(1),"minimum number of CPUs per simulation")
    ("Nmax", po::value<int>(&max_cpus)->default_value(std::numeric_limits<int>::max()),"maximum number of CPUs per simulation")
    ("write-xml","write results to XML files")
    ("input-file", po::value<std::string>(&filename), "input file");
  po::positional_options_description p;
  p.add("input-file", 1);
  
  po::variables_map vm;
  po::store(po::command_line_parser(argc, argv).options(desc).positional(p).run(), vm);
  po::notify(vm);    


  if (vm.count("help")) {
    std::cout << desc << "\n";
    valid=false;
  }
  if (vm.count("license")) {
    print_license(std::cout);
    valid=false;
  }
  if (!valid)
    return;
  
  if (vm.count("mpi")) {
    use_mpi = true;
  }

#ifdef ALPS_HAVE_HDF5
  if (vm.count("write-xml"))
    write_xml = true;
#else
  write_xml=true;
#endif

  if (!filename.empty())
    jobfilename=boost::filesystem::path(filename);
  else
    boost::throw_exception(std::runtime_error("No job file specified"));
    
  if(min_cpus>max_cpus)
    boost::throw_exception(std::runtime_error("Minimum number of CPUs larger than maximum number of CPU"));
  if(min_check_time>max_check_time)
    boost::throw_exception(std::runtime_error("Minimum time between checks larger than maximum time"));
  }
}
    
} // namespace scheduler
} // namespace alps
