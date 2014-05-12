/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@comp-phys.org>,
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
#include <alps/config.h>
#include <cstdio>
#ifdef ALPS_HAVE_UNISTD_H
# include <unistd.h>
#endif

#if defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>
#elif defined(ALPS_HAVE_WINDOWS_H)
# include <windows.h>
#endif

using namespace boost::posix_time;

namespace alps {
namespace scheduler {

MPPScheduler::MPPScheduler(const NoJobfileOptions& opt,const Factory& p)
  : MasterScheduler(opt,p)
{
  if(min_cpus>processes.size())
    boost::throw_exception(std::logic_error("did not get enough processes in MPPScheduler::MPPScheduler"));
}


MPPScheduler::MPPScheduler(const Options& opt,const Factory& p)
  : MasterScheduler(opt,p)
{
  if(min_cpus>processes.size())
    boost::throw_exception(std::logic_error("did not get enough processes in MPPScheduler::MPPScheduler"));
}

int MPPScheduler::run()
{
  ptime end_time=second_clock::local_time()+seconds(long(time_limit));
  unsigned int total_free=0;
  ProcessList free(processes);
  running_tasks=0;

  int all_done=0;

  ptime last_checkpoint=second_clock::local_time();
  if(time_limit>0.)
      std::cout << "Will run " << time_limit << " seconds.\n";
  determine_active();
  assign_processes(free);

  while((!all_done)&&
        (active.size()>0)&&
        ((time_limit<=0)||(second_clock::local_time()<end_time))) {

    if (check_signals() == SignalHandler::TERMINATE) {
      for(unsigned int i=0;i<active.size();i++)
        if(active[i].where.size())
          tasks[active[i].number]->halt();
      checkpoint();
      return -1;
    }              
    
    unsigned int freen=0;
    if(free.size()!=total_free) {
      while(freen!=free.size()) { 
        // check once again, if more processes have become available
        // too expnesive in CPU time on MPP systems
        for (unsigned int i=0;i<active.size();i++)
          check_tasks(free);
        freen=free.size();
      } 
      std::cout  << "Assigning " << free.size() << " processes\n" ;
      assign_processes(free);
    } 
    total_free=free.size();
    all_done=(running_tasks==0);
    
    // do some work on the local simulation
    if(theTask)
      theTask->run();
    else
#if defined(ALPS_HAVE_UNISTD_H)
  sleep(1);    // sleep 1 Sec
#elif defined(ALPS_HAVE_WINDOWS_H)
  Sleep(1000); // sleep 1000 mSec
#else
# error "sleep not found"
#endif

    // check if tasks have finished

    int simfinished = check_tasks(free);
    
    // make checkpoint if necessary
    if((!simfinished)&&(second_clock::local_time()>
        last_checkpoint+seconds(long(checkpoint_time)))&&
       ((time_limit<=0)||(second_clock::local_time()<end_time))) {
        // make regular checkpoints if not yet finished
      std::cout  << "Making regular checkpoint.\n" ;
      last_checkpoint = second_clock::local_time();
      checkpoint();
      std::cout  << "Done with checkpoint.\n" ;
    }
  }
  
  checkpoint();
  if(active.size()) {
    if(all_done)
      std::cout  << "Remaining " << active.size() << " tasks need more than " 
                 << free.size() << " processes.\n";
    else
      std::cout  << "Reached time limit.\n" ;
    return 1;
  }
  else
    std::cout  << "Finished with all tasks.\n" ;
  return 0;
}


// assign free processes to tasks
void MPPScheduler::assign_processes(ProcessList& free)
{
  std::size_t free_processes=free.size();
  while(free_processes&&active.size()) {
    int creation_failed=0;
    std::vector<int> more_processes(active.size());
    unsigned int i;
    for (i=0;i<active.size();i++)
      active[i].work=tasks[active[i].number]->work();

    // first assign processes to largest nonrunning tasks
    int first_new=-1;
    int found=-1;
    std::sort(free.begin(),free.end());
    do {
      found=-1;
      double maxwork=-1;
      for(i=0;i<active.size();i++) {
        if((active[i].where.size()+more_processes[i])==0) {
          double w=active[i].work;
          if(w>maxwork&&free_processes>=min_cpus*active[i].cpus) {
            maxwork=w;
            found=i;
          }
        }
      }
      if(found>=0) {
        if(first_new<0)
          first_new=found;
          more_processes[found]=min_cpus*active[found].cpus;
          free_processes-=min_cpus*active[found].cpus;
        }
      } while ((found >= 0) && free_processes>=min_cpus);

      // don't use local process if no new tasks are started
      if( first_new<0 && free_processes && free[0].local()) {
        std::cout  << "No work for master process\n" ;
        free_processes--;
        free.erase(free.begin());
      }

      // assign remaining processes to the tasks that need them most
      if(free_processes) { // determine work to be done
        unsigned int i;
        do { // look for maximum work simulation
          found=-1;
          double maxwork=-1.;
          for (i=0;i<active.size();i++) {
            if((more_processes[i]||active[i].where.size())&&free_processes>=active[i].cpus)  {
              double w=active[i].work/(active[i].where.size()+more_processes[i]);
              if(w>maxwork&&((max_cpus<=0)||(active[i].where.size()+more_processes[i]/active[i].cpus<max_cpus))) {
                found=i;
                maxwork=w;
              }
            }
          }
              
          // assign one more process to this simulation
          if(found>=0) {
            more_processes[found]+=active[found].cpus;
            free_processes-=active[found].cpus;
          }
        } while (free_processes&&found >=0 );
      }
      
      // now add the processes to the tasks or start the tasks
      // make sure that the local process would be assigned to a new simulation
      ProcessList w;
      ProcessList w1;

      // save loading the local sim for the end, first distribute work
      if(first_new>=0) {
        w1.resize(more_processes[first_new]);
        for (i=0;i<w1.size();i++) {
          w1[i]=*free.begin();
          free.erase(free.begin());
        }
        more_processes[first_new]=0;
      }

      // assign other processors
      for(i=0;i<active.size()&&!creation_failed;i++)
        if(more_processes[i]) {
          w.resize(more_processes[i]);
          std::copy(free.begin(),free.begin()+w.size(),w.begin());
          if(active[i].where.size()) {
            // add more processes
            std::cout  << "Adding " << w.size() << " processes to simulation " << active[i].number+1 << "\n";
            tasks[active[i].number]->add_processes(w);
            active[i].next_check=second_clock::local_time(); // need new estimated
            free.erase(free.begin(),free.begin()+w.size());
          }
        else if(create_task(i,w)) {
          std::cout  << "Created a new simulation: " << active[i].number+1 
                     << " on " <<  more_processes[i] << " processes\n";
          free.erase(free.begin(),free.begin()+more_processes[i]);
        }
        else
          creation_failed = 1;
        w.resize(0);
      }
      
      if(first_new>=0) {
        std::cout << "Creating a new simulation: "
                  << active[first_new].number+1
                  << " on " << w1.size()  << " processes\n";

        if(create_task(first_new,w1))
          more_processes[first_new]=0;
        else {
          std::copy(w1.begin(),w1.end(),std::back_inserter(free));
          creation_failed=1;
        }
        w1.resize(0);
      }

      if((free_processes==free.size())&&(free_processes!=0)) {
        std::cout << "No work for " << free_processes << ".\n";
        free_processes=0;
      }
    } 
  std::cout << "All processes have been assigned\n";
}


// check if tasks have finished
int MPPScheduler::check_tasks(ProcessList& free)
{
  static unsigned int last_check=0;
  int one_finished=0;
  //if(last_check<0)
  //  last_check=0;
  /*
  for (int i=0;i<active.size();i++)
    {
    */
  unsigned int i=last_check;    
  if(i<active.size()) {
    if(active[i].where.size() &&
       second_clock::local_time() > active[i].next_check) {
      double more_time=0.;
      double percentage=0.;
      // if(active[i].next_check==0.)
      //   more_time=-1.;
      int simfinished=tasks[active[i].number]->finished(more_time,percentage);
      // next check after at more_time, restrained to min. and max. times
      more_time=
        (more_time < min_check_time ? min_check_time :
         (more_time > max_check_time ? max_check_time : more_time));
      active[i].next_check=second_clock::local_time()+seconds(long(more_time));
      if(!simfinished)
        std::cout  << "Checking if Simulation " << active[i].number+1
                   << " is finished: "
                   << "not yet, next check in " << int(more_time)
                   << " seconds ( "
                   << static_cast<int>(100.*percentage) << "% done).\n";
      else { 
        std::cout  << "Checking if Simulation " << active[i].number+1
                   << " is finished: "
                   << "Finished\n";
        running_tasks--;
        one_finished=1;
        int j=active[i].number;
        if(theTask == tasks[j])
          theTask=0;
        finish_task(j);
        free.insert(free.end(),active[i].where.begin(),active[i].where.end());
        active.erase(active.begin()+i);
      }
    }
  }
  
  if(!one_finished)
    last_check++;
  if(last_check>=active.size())
    last_check=0;
      
  // put the local process first if it is in the list
  std::sort(free.begin(),free.end());
  return (active.size()==0);
}


// create a new simulation 
int MPPScheduler::create_task(int j,ProcessList& p)
{
  int i=active[j].number;
  if(taskstatus[i]==TaskNotStarted || taskstatus[i]==TaskRunning ||
          (taskstatus[i]==TaskHalted && !tasks[i]->finished_notime())) {
        // create new copy in memory (new start)
      remake_task(p,i);
      if(tasks[i]==0) {
          active.erase(active.begin()+j);
          return 0;
        }
      active[j].where.insert(active[j].where.end(),p.begin(),p.end());
      p.resize(0);
      tasks[i]->start();
      taskstatus[i] = TaskRunning;
      if(tasks[i]->local()) {
          if(theTask)
              boost::throw_exception(std::logic_error( "MPPScheduler::create_simulation: two local tasks"));
          else
            theTask=tasks[i];
        }
      running_tasks++;
      return 1;
    }
  else
    boost::throw_exception( std::logic_error("default reached in MPPScheduler::create_simulation()"));
  return 0;
}

// find tasks that need work
void MPPScheduler::determine_active()
{
  int j=0;
  for(std::size_t i=0;i<tasks.size();i++) {
    if(taskstatus[i]==TaskFinished)
        std::cout  << "Simulation " << i+1 << " finished.\n";
    else if(taskstatus[i]==TaskNotExisting)
        std::cout  << "Simulation " << i+1 << " does not exist.\n";
    else if(taskstatus[i]==TaskNotStarted || taskstatus[i]==TaskRunning ||
            (taskstatus[i]==TaskHalted && !tasks[i]->finished_notime()))
      { // create new copy in memory (new start)
        active.push_back(TaskStatus());
        active[j] = TaskStatus();
        active[j].number = i;        
        active[j].work = tasks[i]->work();
        active[j].cpus= tasks[i]->cpus();
        j++;
      }
    else if (tasks[i]->finished_notime()) 
      finish_task(i);
    else
      boost::throw_exception( std::logic_error( " default reached in MPPScheduler::determine_active()"));
  }
}

} // namespace scheduler
} // namespace alps
