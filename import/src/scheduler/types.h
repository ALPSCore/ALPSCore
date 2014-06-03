/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_SCHEDULER_TYPES_H
#define ALPS_SCHEDULER_TYPES_H

#include <alps/config.h>

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

    // for parallelized runs
    MCDump_run_master              =5,
    MCDump_run_slave               =6,

    // dump version numbers
#ifdef ALPS_ONLY_HDF5
    MCDump_worker_version          =400
#else
    MCDump_worker_version          =310
#endif
    // Some data types changed from 32 to 64 Bit between version 301 and 302
    // vector observable labels stored from 303
    // RNG changed to Mersenne twister from 304
    // minmax dropped in 306
    // went to HDF5 for ALPS-internal data in 310
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
  MCMP_get_observable              = 209,
  MCMP_save_run_to_file            = 211,
  MCMP_load_run_from_file          = 212,
  MCMP_get_run_work                = 215,
  MCMP_set_parameters              = 216,
  MCMP_get_measurements_and_infos  = 217,

// astreich, 06/22
  MCMP_get_summary                 = 220,

// messages returned to the scheduler or task
  MCMP_void                        = 300,
  MCMP_run_dump                    = 301,
  MCMP_run_info                    = 302,
  MCMP_measurements                = 303,
  MCMP_task_finished               = 304,
  MCMP_observable                  = 305,
  MCMP_measurements_and_infos      = 306,
  MCMP_work                        = 311,
  MCMP_run_work                    = 315,
  MCMP_summary                     = 320,

// messages between main and slave runs
  MCMP_do_steps                    = 500
};

} // namespace scheduler
} // namespace alps

#endif // ALPS_SCHEDULER_TYPES_H
