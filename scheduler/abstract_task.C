/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/abstracttask.C   A class to store options
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

#include <alps/scheduler/task.h>
#include <alps/scheduler/types.h>
#include <alps/osiris.h>

namespace alps {

namespace scheduler {

AbstractTask::AbstractTask(const ProcessList& w)
 : where(w)
{
}

AbstractTask::AbstractTask()
{
}

void AbstractTask::add_processes(const ProcessList& p)
{
  if(cpus()==1)
    for (int i=0;i<p.size();i++)
      add_process(p[i]);
  else
    boost::throw_exception(std::logic_error("adding processes to multiple-cpu tasks not yet implemented"));
}

void AbstractTask::delete_processes(const ProcessList& p)
{
  for (int i=0;i<p.size();i++)
      delete_process(p[i]);
}

bool AbstractTask::handle_message(const Process& master,int tag)
{
  IMPDump message;
  OMPDump dump;
  ProcessList pl;
  Process p;
  int32_t n;
  double w;
  std::string filename;
  switch(tag) {
    case MCMP_start_task:
      message.receive(master,MCMP_start_task);
     start();
      return true;
                  
    case MCMP_halt_task:
      message.receive(master,MCMP_halt_task);
      halt();
      return true;
               
    case MCMP_nodes:
      message.receive(master,MCMP_nodes);
      dump << cpus();
      dump.send(master,MCMP_nodes);
      return true;
      
    case MCMP_get_task_finished:
      message.receive(master,MCMP_get_task_finished);
      n = finished(w);
      dump << n << w;
      dump.send(master,MCMP_task_finished);
      return true;
      
    case MCMP_get_work:
      message.receive(master,MCMP_get_work);
      dump << work();
      dump.send(master,MCMP_work);
      return true;
                
    case MCMP_add_processes:
      message.receive(master,MCMP_add_processes);
      message >> pl;
      add_processes(pl);
      break;
  
    case MCMP_add_process:
      message.receive(master,MCMP_add_process);
      message >> p;
      return true;
                    
    case MCMP_delete_processes:
      message.receive(master,MCMP_delete_processes);
      message >> pl;
      delete_processes(pl);
      return true;
  
    case MCMP_delete_process:
      message.receive(master,MCMP_delete_process);
      message >> p;
      delete_process(p);
      return true;
  
    case MCMP_checkpoint:
      message.receive(master,MCMP_checkpoint);
      message >> filename;
      checkpoint(boost::filesystem::path(filename));
      return true;

    default:
      return false;
  }
  return false;
}

} // namespace scheduler

} // namespace alps
