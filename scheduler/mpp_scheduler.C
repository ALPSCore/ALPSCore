/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/mpp_scheduler.C
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/config.h>
#include <alps/osiris.h>
#include <alps/scheduler/scheduler.h>
#include <cstdio>
#ifdef ALPS_HAVE_UNISTD_H
# include <unistd.h>
#endif

namespace alps {
namespace scheduler {

MPPScheduler::MPPScheduler(const Options& opt,const Factory& p)
  : MasterScheduler(opt,p)
{
  if(min_cpus>processes.size())
    boost::throw_exception(std::logic_error("did not get enough processes in MPPScheduler::MPPScheduler"));
}

int MPPScheduler::run()
{
  double end_time=dclock()+time_limit;
  int total_free=0;
  ProcessList free(processes);
  running_tasks=0;

  check_comm_signals();
  int all_done=0;

  double last_checkpoint=dclock();
  if(time_limit>0.)
      std::cout << "Will run " << time_limit << " seconds from now: " << dclock() << "\n";
  determine_active();
  assign_processes(free);

  while((!all_done)&&
	(active.size()>0)&&
	((time_limit<=0)||(dclock()<end_time))) {

    if (check_signals() == SignalHandler::TERMINATE) {
      for(int i=0;i<active.size();i++)
        if(active[i].where.size())
          tasks[active[i].number]->halt();
      checkpoint();
      return -1;
    }	      
    
    check_system(free);
    int freen=0;
    if(free.size()!=total_free) {
      while(freen!=free.size()) { 
        // check once again, if more processes have become available
        // too expnesive in CPU time on MPP systems
        for (int i=0;i<active.size();i++)
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
      sleep(1);

    // check if tasks have finished

    int simfinished = check_tasks(free);
    
    // make checkpoint if necessary
    if((!simfinished)&&(dclock()>last_checkpoint+checkpoint_time)&&
       ((time_limit<=0)||(dclock()<end_time))) {
	// make regular checkpoints if not yet finished
      std::cout  << "Making regular checkpoint.\n" ;
      last_checkpoint = dclock();
      checkpoint();
      std::cout  << "Done with checkpoint.\n" ;
    }
  }
  
  checkpoint();
  if(active.size()) {
    if(all_done)
      std::cout  << "Remaining " << active.size() << " tasks need more than " << free.size()
		 << " processes.\n";
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
  int free_processes=free.size();
  while(free_processes&&active.size()) {
    int creation_failed=0;
    std::vector<int> more_processes(active.size());
    int i;
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
        int i;
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
            active[i].next_check=0.; // need new estimated
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
	std::cout << "Creating new simulation:  " << active[first_new].number+1
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


// check for signals and other events
void MPPScheduler::check_system(ProcessList& /*free*/)
{
/*
  int messageswaiting=1;
  ProcessList changed;
  OMPDump dump;
  int i;
  
  do
    switch(check_comm_signals(changed)) {
      case CommSignal::nosignal:
	messageswaiting=0;
	break;
	
      case CommSignal::host_failed:
      case CommSignal::process_failed:
	// first try to save as much as possible
	if(messageswaiting==1) {
	    checkpoint();
	    messageswaiting = -1;
	  }
	
	// delete all failed processes
        for(i=0;i<active.size();i++)  {
            int k;
            for (k=0;k<changed.size();k++) {
		  Process& p = changed[k];
		  ProcessList::iterator found=
		  std::find(active[i].where.begin(),active[i].where.end(),p);
                  if(found!=active[i].where.end()) {
		      // delete from active list
		      std::cout << "Process " << k << " on sim "
			   << active[i].number << " : " << found << "\n";
		      active[i].where.erase(found);
		      // delete run
		      tasks[active[i].number]->delete_process(p);
		    }
                }
          }

	for (i=0;i<active.size();i++) {
	    int k=active[i].number;
	    if((active[i].where.size()==0)&&(taskstatus[k]==TaskRunning)) {
		std::cout << "Stopping simulation " << k+1 << "\n";
		taskstatus[k] = TaskHalted;
	      }
	  }

        // rebalance the workload
	std::cout << "REBALANCING NOT YET IMPLEMENTED\n";
	break;
	
      case CommSignal::host_added:
        if(runs_parallel()) {
	  TO CHANGE
          dump << dumpname;
          dump.send(changed,MCMP_dump_name);
        }
	free.insert(free.end(), changed.begin(),changed.end());
	break;
	
      default: 
	boost::throw_exception( std::logic_error("default reached in MPPScheduler::check_system()"));
      }
  while (messageswaiting);
  */
}


// check if tasks have finished
int MPPScheduler::check_tasks(ProcessList& free)
{
  static int last_check=0;
  int one_finished=0;
  if(last_check<0)
    last_check=0;
  /*
  for (int i=0;i<active.size();i++)
    {
    */
  int i=last_check;    
  if(i<active.size()) {
      if(active[i].where.size() && dclock() >active[i].next_check) {
	  double more_time=0.;
	  if(active[i].next_check==0.)
	    more_time=-1.;
	  std::cout  << "Checking if Simulation " << active[i].number+1 << " is finished: ";
	  
	  int simfinished=tasks[active[i].number]->finished(more_time);
	  // next check after at more_time, restrained to min. and max. times
	  more_time=
	    (more_time < min_check_time ? min_check_time :
	     (more_time > max_check_time ? max_check_time : more_time));
	  active[i].next_check=dclock()+more_time;
	  if(!simfinished)
	      std::cout  << "Not yet, next check in " << int(more_time) << " seconds.\n";
	  else { 
	      std::cout << "Finished\n";
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
  for(int i=0;i<tasks.size();i++) {
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
