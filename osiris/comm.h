/***************************************************************************
* PALM++/osiris library
*
* osiris/comm.h      communication subroutine header
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
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

#ifndef OSIRIS_COMM_H
#define OSIRIS_COMM_H

// #include <palm/config.h>
#include <alps/osiris/process.h>

#include <cstdlib>
#include <string>
#include <vector>
#include <signal.h>

namespace alps {

//=======================================================================
// INITIALIZATION AND CLEANUP
//
// initialize or stop the message passing library 
//-----------------------------------------------------------------------

// initialize everything

void comm_init(int* argcp, char*** argvp, bool force_master=false);


// stop message passing
// the bool parameter indicates if all slave processes should be killed

void comm_exit(bool kill_slaves=false);


// do we actually run in parallel?

bool runs_parallel();

//=======================================================================
// HOST/PROCESS ENQUIRIES
//
// ask for processes, hosts, ... 
//-----------------------------------------------------------------------

namespace detail {
int local_id(); // return the id of this Process
int invalid_id(); // return an invalid id
}

bool is_master(); // is this the master Process ?

Host local_host(); // make a descriptor of this Host
Process local_process(); // make a descriptor of the local Process
Process process_from_id(const int); // make a descriptor of Process with given id
ProcessList all_processes(); // get a list of all running processes
Process master_process(); // get the master Process

Process start_process(const Host&, const std::string&); // start a Process on the given Host
// start processes on multiple hosts
ProcessList start_processes(const HostList&, const std::string&);
// start as many processes as sensible on all available hosts
ProcessList start_all_processes(const std::string&, unsigned short procs_per_node=1);

} // end namespace alps

#endif // OSIRIS_COMM_H
