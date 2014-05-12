/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*                            Synge Todo <wistaria@comph-phys.org>
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

#include <alps/config.h>
#include <alps/scheduler/scheduler.h>
#include <alps/osiris/mpdump.h>
#if defined(ALPS_HAVE_UNISTD_H)
# include <unistd.h>
#elif defined(ALPS_HAVE_WINDOWS_H)
# include <windows.h>
#endif

namespace alps {
namespace scheduler {

//=======================================================================
// SlaveTask
//
// this class is just a class for communication purposes.
// Member function calls relayed by a MCRemoteRun object are
// received, and the appropriate function is called.
// The return values are sent back.
// Allows for transparent access to remote objects
//-----------------------------------------------------------------------

SlaveTask::SlaveTask(const Process& p)
  : runmaster(p)
{
  theWorker=0;
  started = false;
}        


void SlaveTask::run()
{
  bool messageswaiting=true;
  static IMPDump message;
  do {
    Parameters p;
    ProcessList w;
    int32_t n;
    std::string fname;

    // check for messages and call the appropriate member functions of the run
    int tag=IMPDump::probe(runmaster);
    switch(tag) {
      case 0: // no more messages
        messageswaiting=false;
        break;
          
      case MCMP_make_run:
        message.receive(runmaster,MCMP_make_run);
        if(theWorker)
          boost::throw_exception(std::logic_error("cannot have more than one run per process"));
        message >> w >> p >> n;
        theWorker = theScheduler->make_worker(w,p,n);
        started=false;
        break;
          
      case MCMP_delete_run:
        message.receive(runmaster,MCMP_delete_run);
        if(theWorker) {
          delete theWorker;
          theWorker=0;
        }
        else
          boost::throw_exception(std::logic_error("run does not exist"));
        break;
        
        default:
          messageswaiting= (theWorker ? theWorker->handle_message(runmaster,tag) : false);
        }
    } while (messageswaiting);

  // no more messages: do some work
  if(theWorker) {
    dynamic_cast<Worker&>(*theWorker).run();
  } else {
#if defined(ALPS_HAVE_UNISTD_H)
    sleep(1);    // sleep 1 Sec
#elif defined(ALPS_HAVE_WINDOWS_H)
    Sleep(1000); // sleep 1000 mSec
#else
# error "sleep not found"
#endif
  }
}

void SlaveTask::start()
{
}

void SlaveTask::halt() 
{
}

// OTHER MEMBER FUNCTIONS NEVER USED

void SlaveTask::checkpoint(const boost::filesystem::path&, bool ) const
{
  boost::throw_exception(std::logic_error("should never checkpoint a slave simulation"));
}

void SlaveTask::add_process(const Process& )
{
  boost::throw_exception(std::logic_error("should never add a process to a slave simulation"));
}

double SlaveTask::work() const
{
  boost::throw_exception(std::logic_error("should never obtain work of a slave simulation"));
  return 0.;
}

ResultType SlaveTask::get_summary() const
{
  std::cerr << "\nshould never obtain work of a slave simulation\n\n";
  ResultType res;
  res.count = 0;
  return res;
}
  
bool SlaveTask::finished(double&, double& ) const
{
  boost::throw_exception(std::logic_error("should never ask finished? of a slave simulation"));
  return 0.;
}

uint32_t SlaveTask::cpus() const
{
  boost::throw_exception(std::logic_error("should never get nodes of a slave simulation"));
  return 0;
}

} // namespace scheduler
} // namespace alps
