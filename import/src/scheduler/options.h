/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_SCHEDULER_OPTIONS_H
#define ALPS_SCHEDULER_OPTIONS_H

#include <alps/config.h>
#include <boost/filesystem/path.hpp>
#include <string>

namespace alps {
namespace scheduler {

//=======================================================================
// Options
//
// a class containing the options set by the user, either via command
// line switches or environment variables
//-----------------------------------------------------------------------

class ALPS_DECL NoJobfileOptions
{
public:
  std::string programname;    // name of the executable
  double min_check_time;      // minimum time between checks
  double max_check_time;      // maximum time between checks
  double checkpoint_time;     // time between two checkpoints
  int min_cpus;               // minimum number of runs per simulation
  int max_cpus;               // maximum number of runs per simulation
  double time_limit;          // time limit for the simulation
  bool use_mpi;               // should we use MPI
  bool valid;                 // shall we really run?
  bool write_xml;             // shall we write the results to XML?

  NoJobfileOptions(int argc, char** argv);
  NoJobfileOptions();
};

class Options : public NoJobfileOptions
{
public:
  boost::filesystem::path jobfilename;      // name of the jobfile

  Options(int argc, char** argv);
  Options();
};

} // end namespace
} // end namespace

#endif

