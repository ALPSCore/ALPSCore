/***************************************************************************
* PALM++/osiris library
*
* osiris/comm.h      communication subroutine header
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
