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

#include <alps/scheduler/worker.h>
#include <alps/scheduler/types.h>
#include <alps/scheduler/scheduler.h>
#include <alps/osiris/comm.h>
#include <alps/osiris/mpdump.h>

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

void RemoteWorker::save_to_file(const boost::filesystem::path& fn, const boost::filesystem::path& hdf5fn) const
{ 
  // let the remote process write the run into the file
  OMPDump send;
  send << fn.string () << hdf5fn.string();
  send.send(where,MCMP_save_run_to_file);
}

void RemoteWorker::load_from_file(const boost::filesystem::path& fn, const boost::filesystem::path& hdf5fn)
{
  // let the remote process write the run into the file
  OMPDump send;
  send << fn.string() << hdf5fn.string();
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

// astreich, 06/23
ResultType RemoteWorker::get_summary() const
{
  OMPDump send;
  send.send(where,MCMP_get_summary);
  IMPDump receive(where,MCMP_summary);

  ResultType res;
  receive >> res.T >> res.mean >> res.error >> res.count;
  return res;
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
