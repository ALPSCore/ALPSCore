/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/options.C   A class to store options
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include <alps/scheduler/options.h>

#include <boost/limits.hpp>
#include <boost/throw_exception.hpp>
#include <cstdio>
#include <stdexcept>

namespace alps {
namespace scheduler {

Options::Options(int argc, char** argv) 
  : min_check_time(60.), // don't check more often than once a minute
    max_check_time(900.), // check at least every 15 minutes
    checkpoint_time(1800.), // make checkpoints every 30 minutes
    min_cpus(1), // min # of CPUs per task: default 1
    max_cpus(std::numeric_limits<int>::max()), // max # of CPUs per task: default unlimted
    time_limit(0.) // time limit: unlimited/automatic
{
  // parse all arguments
  if(argc) {
  int i=1;
  while (i < argc) {
    if(argv[i][0]!='-') {
      if (jobfilename.empty())
        jobfilename=boost::filesystem::path(argv[i],boost::filesystem::native);
      else
	boost::throw_exception(std::runtime_error( "Illegal option: " + std::string(argv[i])));
    }
    else {
      if(argv[i][1]=='T') {
        if(i+1<argc) {
          if(!strcmp(argv[i]+2,"c")) {
            if(std::sscanf(argv[++i],"%lf",&checkpoint_time)==EOF)
              boost::throw_exception(std::runtime_error( "illegal checkpoint time"));
          }
          else if(!strcmp(argv[i]+2,"min")) {
            if(std::sscanf(argv[++i],"%lf",&min_check_time)==EOF)
              boost::throw_exception(std::runtime_error( "illegal minimum time"));
          }
          else if(!strcmp(argv[i]+2,"max")) {
	    if(std::sscanf(argv[++i],"%lf",&max_check_time)==EOF)
              boost::throw_exception(std::runtime_error( "illegal maximum time"));
          }
          else if(argv[i][2]=='\0') {
	    if(std::sscanf(argv[++i],"%lf",&time_limit)==EOF)
              boost::throw_exception(std::runtime_error( "illegal time limit"));
          }
          else
            boost::throw_exception(std::runtime_error("Illegal option"+std::string (argv[i])));
        }
        else
          boost::throw_exception(std::runtime_error( "argument to last option missing"));
        }
      else if(argv[i][1]=='N') {
        if(i+1<argc) {
          if(!strcmp(argv[i]+2,"min")) {
            if(std::sscanf(argv[++i],"%d",&min_cpus)==EOF)
              boost::throw_exception(std::runtime_error( "illegal CPU number"));
           }
          else if(!strcmp(argv[i]+2,"max")) {
            if(std::sscanf(argv[++i],"%d",&max_cpus)==EOF)
              boost::throw_exception(std::runtime_error( "illegal CPU number"));
          }
          else 
            boost::throw_exception(std::runtime_error( "Illegal option: " + std::string(argv[i])));
        }
        else
          boost::throw_exception(std::runtime_error( "argument to last option missing"));
      }
      else
        boost::throw_exception(std::runtime_error( "Illegal option: " + std::string(argv[i])));
    }      
      i++; // next option
  }
  
  // store the program name:
  
  programname = argv[0];
  if(min_cpus>max_cpus)
    boost::throw_exception(std::runtime_error("Minimum number of CPUs larger than maximum number of CPU"));
  if(min_check_time>max_check_time)
    boost::throw_exception(std::runtime_error("Minimum time between checke larger than maximum time"));
  if(jobfilename.empty())
    boost::throw_exception(std::runtime_error("No job file specified"));
  }
}

} // namespace scheduler
} // namespace alps
