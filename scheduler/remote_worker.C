/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/remote_worker.C   A class to store parameters
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

#include <alps/scheduler/worker.h>
#include <alps/scheduler/types.h>
#include <alps/scheduler/scheduler.h>
#include <alps/osiris.h>

namespace alps {
namespace scheduler {

//=======================================================================
// This file defines the classes which handles access to
// a run on a remote process
//=======================================================================

RemoteWorker::RemoteWorker(const ProcessList& w, const alps::Parameters& p,int32_t n)
  : AbstractWorker(),
    where(w[n])
{
  // send a run creation message containg the parameters
  Scheduler::make_slave_task(where);

  OMPDump dump;
  dump.init();
  dump << w;
  dump << p;
  dump << n;
  dump.send(where,MCMP_make_run);
}

// send run destruction message to remote process
RemoteWorker::~RemoteWorker()
{
  OMPDump dump;
  dump.send(where,MCMP_delete_run);
  Scheduler::delete_slave_task(where);
}

void RemoteWorker::save_to_file(const boost::filesystem::path& fn) const
{
  // let the remote process write the run into the file
  OMPDump send;
  send << fn.string ();
  send.send(where,MCMP_save_run_to_file);
}

void RemoteWorker::load_from_file(const boost::filesystem::path& fn)
{
  // let the remote process write the run into the file
  OMPDump send;
  send << fn.string();
  send.send(where,MCMP_load_run_from_file);
}

void RemoteWorker::set_parameters(const alps::Parameters& p)
{
  // let the remote process write the run into the file
  OMPDump send;
  send << p;
  send.send(where,MCMP_set_parameters);
}

void RemoteWorker::start_worker()
{
  OMPDump dump;
  dump.send(where,MCMP_startRun);
}

void RemoteWorker::halt_worker()
{
  OMPDump dump;
  dump.send(where,MCMP_haltRun);
}


double RemoteWorker::work_done() const
{
  // send message to remote process
  OMPDump send;
  send.send(where,MCMP_get_run_work);
  IMPDump receive(where,MCMP_run_work);
  
  // load measurements from message
  double ww;
  receive >> ww;
  return ww;
}


TaskInfo RemoteWorker::get_info() const
{
  // send message to remote process
  OMPDump send;
  send.send(where,MCMP_get_run_info);
  IMPDump receive(where,MCMP_run_info);
  
  // load measurements from message
  TaskInfo info;
  receive >> info;
  return info;
}

bool RemoteWorker::handle_message(const Process& ,int32_t)
{
  boost::throw_exception(std::logic_error("RemoteWorker should never handle a message"));
  return true;
}

} // namespace scheduler
} // namespace alps
