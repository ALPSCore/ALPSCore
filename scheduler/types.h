/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/types.h
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

#ifndef ALPS_SCHEDULER_FORWARD_H___
#define ALPS_SCHEDULER_FORWARD_H___

//=======================================================================
// This file constants such as magic ids for dumps and message ids
//=======================================================================

namespace alps {
namespace scheduler {

// dump types

enum MCDumpType {
    // dump magic numbers
    MCDump_scheduler		=1,
    MCDump_task              =2,
    MCDump_run                     =3,
    MCDump_measurements            =4,

    // dump version numbers
    MCDump_task_version      =200,
    MCDump_run_version             =200,
    MCDump_measurements_version    =100
};


//=======================================================================
// message tags
//-----------------------------------------------------------------------

enum MCMP_Tags {
// messages sent to the slave scheduler by the master
  MCMP_stop_slave_scheduler        = 101,
  MCMP_make_slave_task             = 102,
  MCMP_make_task                   = 103,
  MCMP_dump_name		   = 104,
  MCMP_delete_task                 = 106,
  MCMP_get_task_finished           = 108,
  MCMP_start_task                  = 109,
  MCMP_halt_task                   = 110,
  MCMP_add_processes               = 114,
  MCMP_add_process                 = 115,
  MCMP_delete_processes            = 116,
  MCMP_delete_process              = 117,
  MCMP_checkpoint                  = 118,
  MCMP_get_work                    = 119,
  MCMP_nodes                       = 122,

// messages sent to the slave task by the task
  MCMP_make_run                    = 201,
  MCMP_startRun                    = 203,
  MCMP_haltRun                     = 204,
  MCMP_delete_run                  = 206,
  MCMP_get_run_info                = 207,
  MCMP_get_measurements            = 208,
  MCMP_save_run_to_file            = 211,
  MCMP_load_run_from_file          = 212,
  MCMP_get_run_work		   = 215,
  MCMP_set_parameters              = 216,

// messages returned to the scheduler or task
  MCMP_void                        = 300,
  MCMP_run_dump                    = 301,
  MCMP_run_info                    = 302,
  MCMP_measurements                = 303,
  MCMP_task_finished               = 304,
  MCMP_work                        = 311,
  MCMP_run_work			   = 315,

// messages between main and slave runs
  MCMP_do_steps                    = 500
};

} // namespace scheduler
} // namespace alps

#endif
