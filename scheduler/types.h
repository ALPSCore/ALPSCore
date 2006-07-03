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
    MCDump_scheduler               =1,
    MCDump_task                    =2,
    MCDump_run                     =3,
    MCDump_measurements            =4,

    // dump version numbers
    MCDump_worker_version          =303
    // Some data types changed from 32 to 64 Bit between version 301 and 302
    // vector observable labels stored from 303
};


//=======================================================================
// message tags
//-----------------------------------------------------------------------

enum MCMP_Tags {
// messages sent to the slave scheduler by the master
  MCMP_stop_slave_scheduler        = 101,
  MCMP_make_slave_task             = 102,
  MCMP_make_task                   = 103,
  MCMP_dump_name                   = 104,
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
// astreich, inserted for sychronisation
  MCMP_ready                       = 150,

// messages sent to the slave task by the task
  MCMP_make_run                    = 201,
  MCMP_startRun                    = 203,
  MCMP_haltRun                     = 204,
  MCMP_delete_run                  = 206,
  MCMP_get_run_info                = 207,
  MCMP_get_measurements            = 208,
  MCMP_save_run_to_file            = 211,
  MCMP_load_run_from_file          = 212,
  MCMP_get_run_work                = 215,
  MCMP_set_parameters              = 216,
// astreich, 06/22
  MCMP_get_summary                 = 220,

// messages returned to the scheduler or task
  MCMP_void                        = 300,
  MCMP_run_dump                    = 301,
  MCMP_run_info                    = 302,
  MCMP_measurements                = 303,
  MCMP_task_finished               = 304,
  MCMP_work                        = 311,
  MCMP_run_work                    = 315,
  MCMP_summary                     = 320,

// messages between main and slave runs
  MCMP_do_steps                    = 500
};

} // namespace scheduler
} // namespace alps

#endif
