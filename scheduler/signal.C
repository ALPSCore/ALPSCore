/***************************************************************************
* ALPS++/alps library
*
* alps/signal.C      dumps for object serialization
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
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

// TODO: signals sent by a message
 
#include <alps/scheduler/signal.hpp>

#include <boost/throw_exception.hpp>
#include <iostream>
#include <signal.h>
#include <stdexcept>
#ifdef HAVE_UNISTD_H
# include <unistd.h>
#endif
#ifdef HAVE_SYS_TYPES_H
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
	  
      // register the signal handlers
      signal(SIGINT, reinterpret_cast<signal_handle_ptr>(&kill));
      signal(SIGTERM, reinterpret_cast<signal_handle_ptr>(&kill));
      signal(SIGQUIT, reinterpret_cast<signal_handle_ptr>(&kill));
      signal(SIGTSTP, reinterpret_cast<signal_handle_ptr>(&tstp));
      signal(SIGUSR1, reinterpret_cast<signal_handle_ptr>(&usr1));
      signal(SIGUSR1, reinterpret_cast<signal_handle_ptr>(&usr2));
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
  ::kill(getpid(),SIGSTOP); // stop myself
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
