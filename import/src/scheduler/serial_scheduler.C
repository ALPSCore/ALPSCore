/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

namespace alps {
namespace scheduler {

//=======================================================================
// SerialScheduler
// 
// a scheduler for a single CPU, finishes one task after the other
// on multi-node computers or clusters all nodes work for the same
// Task
//-----------------------------------------------------------------------

using namespace boost::posix_time;

SerialScheduler::SerialScheduler(const NoJobfileOptions& opt,const Factory& p)
  : MasterScheduler(opt,p)
{
  if(min_cpus!=1)
    boost::throw_exception(std::invalid_argument("cannot do more than one run on a single process"
));
}


SerialScheduler::SerialScheduler(const Options& opt,const Factory& p)
  : MasterScheduler(opt,p)
{
  if(min_cpus!=1)
    boost::throw_exception(std::invalid_argument("cannot do more than one run on a single process"));
 }

int SerialScheduler::run()
{
  ptime end_time=second_clock::local_time()+seconds(long(time_limit));
  ptime task_time(second_clock::local_time());
  // do all Tasks
  for(unsigned int i=0;i<tasks.size();i++) {
    if(time_limit>0. && second_clock::local_time()>end_time)
      return 1;
    if(taskstatus[i]==TaskFinished)
      std::cout << "Task " << i+1 << " finished.\n";
    else if(taskstatus[i]==TaskNotExisting)
      std::cout  << "Task " << i+1 << " does not exist.\n";
    else if(taskstatus[i]==TaskNotStarted || taskstatus[i]==TaskRunning ||
            (taskstatus[i]==TaskHalted)) {
      unsigned int n=tasks[i]->cpus();
      if(n<1)
        boost::throw_exception(std::logic_error("at least one node required for a run!"));
      if(n>processes.size())
        std::cout  << "Task " << i+1 << " needs more nodes than available and will not run.\n";
      else {
        // create new Task in memory (new start)
        remake_task(processes,i);
        theTask=tasks[i];
      }
    }
    else
      boost::throw_exception( std::logic_error("In SerialScheduler::start: illegal Task status"));
    if(taskstatus[i]==TaskNotStarted || taskstatus[i]==TaskRunning || (taskstatus[i]==TaskHalted)) { 
      // do work with this Task
      taskstatus[i] = TaskRunning;
      std::cout  << "Starting task " << i+1 << ".\n";
      tasks[i]->start();

      task_time = second_clock::local_time();
      int task_finished=0;
      ptime next_check=second_clock::local_time();
      ptime last_checkpoint=second_clock::local_time();
        
      do {
        if (check_signals() == SignalHandler::TERMINATE) {
          theTask->halt();
          checkpoint();
          return -1;
        }              
         
        theTask->run();
          
        if(time_limit >0. && second_clock::local_time()>end_time) {
          std::cout << "Time limit exceeded\n";
          if (theTask->finished_notime())
            finish_task(i);
          else
            theTask->halt();
          checkpoint();
          return 1;
        }
          
        if(second_clock::local_time()>next_check) {
          std::cout  << "Checking if it is finished: " << std::flush;
          double more_time=0;
          double percentage=0.;
          task_finished=theTask->finished(more_time,percentage);
              
          // next check after at more_time, restrained to min. and max. times
          more_time= (more_time < min_check_time ? min_check_time :
                      (more_time > max_check_time ? max_check_time : more_time));
          next_check=second_clock::local_time()+seconds(int(more_time));
          std::cout  << "not yet, next check in " << int(more_time) << " seconds ( "
            << static_cast<int>(100.*percentage) << "% done).\n";
        }
        if((!task_finished)&&(second_clock::local_time()>last_checkpoint+seconds(int(checkpoint_time)))) {
          // make regular checkpoints if not yet finished
          std::cout  << "Making regular checkpoint.\n";
          checkpoint();
          last_checkpoint=second_clock::local_time();
          std::cout  << "Done with checkpoint.\n";
        }
      } while (!task_finished);
            
      finish_task(i);
      std::cout  << "This task took " << (second_clock::local_time()-task_time).total_seconds() << " seconds.\n";
      checkpoint();
    }
  }
  std::cout << "Finished with everything.\n";
  return 0;
}

} // namespace scheduler
} // namespace alps
