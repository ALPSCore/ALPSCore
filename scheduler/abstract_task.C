/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#include <boost/iterator/iterator_adaptor.hpp>
#include <alps/scheduler/task.h>
#include <alps/scheduler/types.h>
#include <alps/osiris/mpdump.h>
#include <alps/osiris/std/string.h>

namespace alps {

namespace scheduler {

AbstractTask::AbstractTask(const ProcessList& w)
 : where(w)
{ 
  use_error_limit=false;
}

AbstractTask::AbstractTask()
{
  use_error_limit=false;
}

AbstractTask::~AbstractTask()
{
}


void AbstractTask::add_processes(const ProcessList& p)
{
  if(cpus()==1)
    for (std::size_t i=0;i<p.size();i++)
      add_process(p[i]);
  else
    boost::throw_exception(std::logic_error("adding processes to multiple-cpu tasks not yet implemented"));
}

bool AbstractTask::handle_message(const Process& master,int tag)
{
  IMPDump message;
  OMPDump dump;
  ProcessList pl;
  Process p;
  int32_t n;
  double w=0.;
  bool flag;
  double percentage=0.;
  std::string filename;
  ResultType res;
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
      n = finished(w,percentage);
      dump << n << w << percentage;
      dump.send(master,MCMP_task_finished);
      return true;
      
    case MCMP_get_work:
      message.receive(master,MCMP_get_work);
      dump << work();
      dump.send(master,MCMP_work);
      return true;

    case MCMP_get_summary:
      // return a summary to the master
      message.receive(master,MCMP_get_summary);
      res = get_summary();
      dump << res.T << res.mean << res.error << res.count;
      dump.send(master, MCMP_summary);
      break;

    case MCMP_add_processes:
      message.receive(master,MCMP_add_processes);
      message >> pl;
      add_processes(pl);
      break;
  
    case MCMP_add_process:
      message.receive(master,MCMP_add_process);
      message >> p;
      return true;
                    
    case MCMP_checkpoint:
      message.receive(master,MCMP_checkpoint);
      message >> filename >> flag;
      checkpoint(boost::filesystem::path(filename),flag);
      return true;

    default:
      return false;
  }
  return false;
}

} // namespace scheduler

} // namespace alps
