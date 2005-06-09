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

#include <boost/iterator/iterator_adaptor.hpp>
#include <alps/scheduler/task.h>
#include <alps/scheduler/types.h>
#include <alps/osiris.h>

namespace alps {

namespace scheduler {

AbstractTask::AbstractTask(const ProcessList& w)
 : where(w)
{ 
  /* astreich, 05/25 */
  use_error_limit=false;
}

AbstractTask::AbstractTask()
{
  /* astreich, 05/25 */
  use_error_limit=false;
}
void AbstractTask::setErrorLimit(std::string name, double value) {
  obs_name_for_limit = name;
  error_limit = value;
  use_error_limit = true;
  // set error limit to worker
  if (theWorker)    
    theWorker->setErrorLimit(name,value);
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
