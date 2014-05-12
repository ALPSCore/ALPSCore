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

// This file implements behavior common to all schedulers

#include <alps/scheduler/scheduler.h>
#include <alps/scheduler/types.h>
#include <alps/parameter.h>
#include <alps/config.h>
#include <alps/utility/copyright.hpp>
#include <alps/osiris/comm.h>
#include <alps/osiris/mpdump.h>

#ifdef ALPS_HAVE_UNISTD_H
# include <unistd.h>
#elif defined(ALPS_HAVE_WINDOWS_H)
# include <windows.h>
#endif


namespace alps {
namespace scheduler {

void print_copyright(std::ostream& out) {
  out << "using the ALPS parallelizing scheduler\n";
  out << "  copyright (c) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>.\n";
  out << "  see Lecture Notes in Computer Science, Vol. 1505, p. 191 (1998).\n\n";
}

// global variable: the scheduler on this node
Scheduler* theScheduler=0;

//=======================================================================
// Scheduler
//
// the base class for schedulers, defining common functions
//-----------------------------------------------------------------------

int Scheduler::run() // a slave scheduler
{
  bool terminate = false;
  bool messageswaiting=false;
  
  Process master=master_process();
  Process simmaster;
  std::string filename;
  ProcessList where;
  IMPDump message;
  Parameters param;
  
  do {
      // check true
      messageswaiting=true;
      do {
        if(IMPDump::probe(master,MCMP_stop_slave_scheduler)) {
          message.receive(master,MCMP_stop_slave_scheduler);
          terminate=true;
        }
      OMPDump dump;
      if(!theTask) {
        switch(IMPDump::probe()) {
          case 0:
            messageswaiting=false; // no messages
            break;
                  
          case MCMP_stop_slave_scheduler:
            break; // dealt with at another place
                  
          case MCMP_make_slave_task:
            message.receive(MCMP_make_slave_task);
            simmaster = message.sender();
            theTask = new SlaveTask(simmaster);
            break;
                  
          case MCMP_make_task:
            message.receive(MCMP_make_task);
            simmaster = message.sender();
            message >> where >> filename;
            theTask = make_task(where,boost::filesystem::path(filename));
            break;
         
          case MCMP_ready:
            // do nothing
            break;
 
          default:
            boost::throw_exception( std::logic_error("received invalid message in Scheduler::run()"));
        }                
      }
      else {
        int tag=IMPDump::probe(simmaster);
        switch(tag) {
          case 0: // no messages
            messageswaiting=false;
            break;
                    
          case MCMP_delete_task:
            message.receive(simmaster,MCMP_delete_task);
            delete theTask;
            theTask=0;
            simmaster = Process();
            break;
                
          default:
            messageswaiting=theTask->handle_message(simmaster,tag);
            break;
        }
      }
    } while (messageswaiting);
    
    if(terminate&&!theTask)
      return 0; // received stop message and all other messages are processed
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
  } while(true) ;// forever
}

Scheduler::Scheduler(const NoJobfileOptions& opt, const Factory& p)
  : proc(p), 
    programname(opt.programname), 
    theTask(0),
    min_check_time(opt.min_check_time),
    max_check_time(opt.max_check_time),
    checkpoint_time(opt.checkpoint_time),
    min_cpus(opt.min_cpus),
    max_cpus(opt.max_cpus),
    time_limit(opt.time_limit),
    write_xml(opt.write_xml)
{
  processes = all_processes();
  theScheduler=this;
  use_error_limit = false;
  make_summary = false;
}


// load/create tasks and runs

// creation of slave task by message to slave scheduler
void Scheduler::make_slave_task(const Process& w)
{
  OMPDump dump;
  dump.send(w,MCMP_make_slave_task);
}

// similar deletion of slave task
void Scheduler::delete_slave_task(const Process& w)
{
  OMPDump dump;
  dump.send(w,MCMP_delete_task);
}

// load from file
AbstractTask* Scheduler::make_task(const ProcessList& w,const boost::filesystem::path& fname)
{
  ProcessList where(w); // we will modify it
  bool found=false;
  
  if(where.empty())
    found=true; // just information, start locally
  else { //look for the local process
    std::sort(where.begin(),where.end());
    found=where.begin()->local();
  }
  if(found)  { // local
    AbstractTask* task= proc.make_task(where,fname);
    //dynamic_cast<Task*>(task)->construct();

    return task;
  }
  else
    return new RemoteTask(where,fname);

}

AbstractTask* Scheduler::make_task(const boost::filesystem::path& fn)
{
  ProcessList nowhere;
  return make_task(nowhere,fn);
}

AbstractWorker* Scheduler::make_worker(const ProcessList& w, const alps::Parameters& p,int n)
{
  return proc.make_worker(w,p,n);
}

AbstractWorker* Scheduler::make_worker(const alps::Parameters& p)
{
  return proc.make_worker(ProcessList(),p,0);
}

void init(const Factory& p)
{
  theScheduler = new SerialScheduler(Options(),p);
}

// initialize a scheduler for real work, parsing the command line
int start(int argc, char** argv, const Factory& p)
{
  Options opt(argc,argv);
  comm_init(argc,argv,opt.use_mpi);
  if (is_master() || !runs_parallel()) {
    p.print_copyright(std::cout);
    alps::scheduler::print_copyright(std::cout);
    alps::print_copyright(std::cout);
  }
  
  int res=0;
  if (opt.valid) {
    if(!runs_parallel()) 
      theScheduler = new SerialScheduler(opt,p);
    else if (is_master()) 
      theScheduler = new MPPScheduler(opt,p);
    else 
      theScheduler = new Scheduler(opt,p);
    res = theScheduler->run();
    delete theScheduler; 
  }
  comm_exit();
  return res;
}

void Scheduler::set_time_limit(double limit)
{
  time_limit = limit;
}

void Scheduler::checkpoint()
{
  // nothing done by default
}



int Scheduler::check_signals()
{
  switch(sig())
    {
    case SignalHandler::NOSIGNAL:
        break;
      
    case SignalHandler::USER1:
    case SignalHandler::USER2:
      std::cout << "Checkpointing...\n";
      checkpoint();
      break;
      
    case SignalHandler::STOP:
      std::cout  << "Checkpointing and stopping...\n";
      checkpoint();
      sig.stopprocess(); // stop the process
      break;

    case SignalHandler::TERMINATE:
      std::cout  << "Checkpointing and exiting...\n";
      return SignalHandler::TERMINATE;
      
    default:
      boost::throw_exception ( std::logic_error( "default on switch reached in MasterScheduler::check_signals()"));
    }
  return SignalHandler::NOSIGNAL;
}

Scheduler::~Scheduler()
{
  if(is_master() && processes.size()>1) 
  {
    OMPDump dump;
    dump.send(processes,MCMP_stop_slave_scheduler);
  }
}

} // namespace scheduler
} // namespace alps
