/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/remote_task.C
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

#include <alps/scheduler/scheduler.h>
#include <alps/osiris.h>

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

void RemoteTask::delete_processes(const ProcessList& p)
{
  ProcessList::iterator found;
  
  bool found_one=false;
  bool found_master=false;
  for (int i=0;i<p.size();i++)
      {
	found=std::find(where.begin(),where.end(),p[i]);
	if(found==where.begin())
	  found_master=true;
	if(found != where.end()) {
	  where.erase(found);
	  found_one=true;
	}
      }
  if(found_one && ! found_master)
    {
      OMPDump send;
      send << p;
      send.send(where[0],MCMP_delete_processes);
    }
}

void RemoteTask::delete_process(const Process& p)
{
  ProcessList::iterator found = std::find(where.begin(),where.end(),p);
  if(found != where.end()) {
    bool is_start = (found == where.begin());
    where.erase(found);
    if(!is_start) {
      OMPDump send;
      send << p;
      send.send(where[0],MCMP_delete_process);
    }
  }
}

bool RemoteTask::finished(double& more_time) const
{
  OMPDump send;
  send.send(where[0],MCMP_get_task_finished);
      
  IMPDump receive(where[0],MCMP_task_finished);
      
  int32_t flag;
  receive >> flag;
  receive >> more_time;
  return flag;
}

double RemoteTask::work() const
{
  OMPDump send;
  send.send(where[0],MCMP_get_work);
  IMPDump receive(where[0],MCMP_work);
  return static_cast<double>(receive);
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

void RemoteTask::checkpoint(const boost::filesystem::path& fn) const
{
  OMPDump send;
  send << fn.string();
  send.send(where[0],MCMP_checkpoint);
}

bool RemoteTask::handle_message(const Process& ,int )
{
  boost::throw_exception(std::logic_error("RemoteTask should never handle a message"));
  return true;
}

} // namespace scheduler
} // namespace alps
