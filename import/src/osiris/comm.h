/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

#ifndef OSIRIS_COMM_H
#define OSIRIS_COMM_H

#include <alps/osiris/process.h>
#include <alps/config.h>

namespace alps {


//=======================================================================
// INITIALIZATION AND CLEANUP
//
// initialize or stop the message passing library 
//-----------------------------------------------------------------------

// initialize everything

ALPS_DECL void comm_init(int& argc, char**& argv, bool=false);


// stop message passing
// the bool parameter indicates if all slave processes should be killed

ALPS_DECL void comm_exit(bool kill_slaves=false);


// do we actually run in parallel?

ALPS_DECL bool runs_parallel();

//=======================================================================
// HOST/PROCESS ENQUIRIES
//
// ask for processes, hosts, ... 
//-----------------------------------------------------------------------

namespace detail {
int local_id(); // return the id of this Process
int invalid_id(); // return an invalid id
}

ALPS_DECL bool is_master(); // is this the master Process ?

Process local_process(); // make a descriptor of the local Process
ProcessList all_processes(); // get a list of all running processes
Process master_process(); // get the master Process

} // end namespace alps

#endif // OSIRIS_COMM_H
