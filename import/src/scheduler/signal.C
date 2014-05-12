/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

// TODO: signals sent by a message
 
#include <alps/scheduler/signal.hpp>

#include <boost/throw_exception.hpp>
#include <iostream>
#include <signal.h>
#include <stdexcept>
#ifdef ALPS_HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef ALPS_HAVE_SYS_TYPES_H
# include <sys/types.h>
#endif

namespace alps {
namespace scheduler {

//=======================================================================
// static member objects
//-----------------------------------------------------------------------

unsigned int SignalHandler::u1; // number of user1 signals received
unsigned int SignalHandler::u2; // number of user2 signals received
unsigned int SignalHandler::k; // number of terminate signals received
unsigned int SignalHandler::s; // number of stop signals received
unsigned int SignalHandler::count; // total number of signals received
bool SignalHandler::initialized=false; 

extern "C" {
  typedef void (*signal_handle_ptr)(int i);
}

SignalHandler::SignalHandler()
{
  if(!initialized)
    {
      initialized=true;
      u1=u2=k=s=count=0;

#ifndef BOOST_MSVC
      // register the signal handlers
      struct sigaction Action;
      sigemptyset(&Action.sa_mask);
      // restart interrupted system calls, and make the signals one-shot
      Action.sa_flags = SA_RESTART | SA_RESETHAND; 
          
      Action.sa_handler = reinterpret_cast<signal_handle_ptr>(&kill);
      sigaction(SIGINT, &Action, NULL);
      sigaction(SIGTERM, &Action, NULL);
      sigaction(SIGQUIT, &Action, NULL);

      // SIGTSTP handler disabled for now, it seems to be not so useful
      //Action.sa_handler = reinterpret_cast<signal_handle_ptr>(&tstp);
      //sigaction(SIGTSTP, &Action, NULL);

      Action.sa_handler = reinterpret_cast<signal_handle_ptr>(&usr1);
      sigaction(SIGUSR1, &Action, NULL);
      Action.sa_handler = reinterpret_cast<signal_handle_ptr>(&usr2);
      sigaction(SIGUSR2, &Action, NULL);
#else
      // ToDo: register signal handers under Windows environment (How?)
#endif
    }
}


// functions to be called if signals are intercepted

void SignalHandler::kill(int)
{
  if(!k)
    {
      count++;
      k++;
    }
}

void SignalHandler::usr1(int)
{
  if(!u1)
    {
      count++;
      u1++;
    }
}


void SignalHandler::usr2(int)
{
  if(!u2)
    {
      count++;
      u2++;
    }
}


void SignalHandler::tstp(int)
{
  if(!s)
    {
      count++;
      s++;
    }
}


// stop this Process

void SignalHandler::stopprocess()
{
#ifndef BOOST_MSVC
  ::kill(getpid(),SIGSTOP); // stop myself
#endif
}  


// check for a signal

SignalHandler::SignalInfo SignalHandler::operator() ()
{
  if(!count)
    {
        return NOSIGNAL;
     }
  
  if(u1)
    {
      count--;
      u1--;
      return USER1; // has highest priority
    }
    
  if(u2)
    {
      count--;
      u2--;
      return USER2; // has 2nd highest priority
    }

  if(s)
    {
      count--;
      s--;
      return STOP; // 3rd highest priority
    }
    
  if(k)
     {
      count--;
      k--;
      return TERMINATE; // only after all other signals have been processed
    }
    
  boost::throw_exception(std::logic_error("total number of signals does not match sum in SignalHandler"));
  return TERMINATE;
}

} // namespace scheduler
} // namespace alps
