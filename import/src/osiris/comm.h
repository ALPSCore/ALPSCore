/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
