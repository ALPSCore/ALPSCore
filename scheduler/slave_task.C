/***************************************************************************
* ALPS++/scheduler library
*
* scheduler/slave_task.C   A class to store parameters
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
**************************************************************************/

#include <alps/config.h>
#include <alps/osiris.h>
#include <alps/scheduler/scheduler.h>
#ifdef ALPS_HAVE_UNISTD_H
# include <unistd.h>
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
  if(theWorker)
    dynamic_cast<Worker&>(*theWorker).run();
  else
    sleep(1);
}

void SlaveTask::start()
{
}

void SlaveTask::halt() 
{
}

// OTHER MEMBER FUNCTIONS NEVER USED

void SlaveTask::checkpoint(const boost::filesystem::path& ) const
{
  boost::throw_exception(std::logic_error("should never checkpoint a slave simulation"));
}

void SlaveTask::add_process(const Process& )
{
  boost::throw_exception(std::logic_error("should never add a process to a slave simulation"));
}

void SlaveTask::delete_process(const Process&)
{
  boost::throw_exception(std::logic_error("should never delete a process from a slave simulation"));
}

double SlaveTask::work() const
{
  boost::throw_exception(std::logic_error("should never obtain work of a slave simulation"));
  return 0.;
}

bool SlaveTask::finished(double&) const
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
