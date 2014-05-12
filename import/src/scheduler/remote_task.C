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

#include <alps/scheduler/scheduler.h>
#include <alps/osiris/mpdump.h>
#include <alps/osiris/std/string.h>

namespace alps {
namespace scheduler {

RemoteTask::RemoteTask(const ProcessList& w, const boost::filesystem::path& fn)
 : AbstractTask(w)
{
  OMPDump message;
  message << w;
  message << fn.string();
  message.send(where[0],MCMP_make_task);
}

RemoteTask::~RemoteTask()
{
      OMPDump message;
      message.send(where[0],MCMP_delete_task);
}

void RemoteTask::add_processes(const ProcessList& p)
{
      OMPDump send;
      send << p;
      send.send(where[0],MCMP_add_processes);
}

void RemoteTask::add_process(const Process& p)
{
      OMPDump send;
      p.save(send);
      send.send(where[0],MCMP_add_process);
}

bool RemoteTask::finished(double& more_time, double& percentage) const
{
  OMPDump send;
  send.send(where[0],MCMP_get_task_finished);
      
  IMPDump receive(where[0],MCMP_task_finished);
      
  int32_t flag;
  receive >> flag >> more_time >> percentage;
  return flag!=0;
}

double RemoteTask::work() const
{
  OMPDump send;
  send.send(where[0],MCMP_get_work);
  IMPDump receive(where[0],MCMP_work);
  return static_cast<double>(receive);
}

/**
 * Sends the summary for this remote task to the master
 */
ResultType RemoteTask::get_summary() const
{
  OMPDump send;
  send.send(where[0],MCMP_get_summary);
  IMPDump receive(where[0],MCMP_summary);
  ResultType res;
  receive >> res.T;
  receive >> res.mean;
  receive >> res.error;
  receive >> res.count;
  return res;
}

void RemoteTask::run()
{
  boost::throw_exception(std::logic_error("RemoteTask::run should never be called"));
}
        
void RemoteTask::start()
{
  OMPDump dump;
  dump.send(where[0],MCMP_start_task);
}

void RemoteTask::halt()
{
  OMPDump dump;
  dump.send(where[0],MCMP_halt_task);
}

uint32_t RemoteTask::cpus() const
{
  OMPDump send;
  send.send(where,MCMP_nodes);
  IMPDump receive(where[0],MCMP_nodes);
  return static_cast<uint32_t>(receive);
}

void RemoteTask::checkpoint(const boost::filesystem::path& fn,bool write_all_xml) const
{
  OMPDump send;
  send << fn.string() << write_all_xml;
  send.send(where[0],MCMP_checkpoint);
}

bool RemoteTask::handle_message(const Process& ,int )
{
  boost::throw_exception(std::logic_error("RemoteTask should never handle a message"));
  return true;
}

} // namespace scheduler
} // namespace alps
