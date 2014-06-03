/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_SIGNAL_H
#define ALPS_SIGNAL_H

//=======================================================================
// This file defines a signal handler class
//=======================================================================

#include <alps/config.h>

namespace alps {
namespace scheduler {

/** implements a signal handler.
    signals are intercepted and can be checked for.
*/

class ALPS_DECL SignalHandler
{
public:
  /** symbolic names for signals.
    SIGINT, SIGQUIT and SIGTERM are mapped to TERMINATE
    SIGTSTP is mapped to STOP
    SIGUSR1 is mapped to USER1
    SIGUSR2 is mapped to USER2*/

    enum SignalInfo
    {
      NOSIGNAL=0,
      USER1,
      USER2,
      STOP,
      TERMINATE
    };

  /// a default constructor
  SignalHandler();

  /** ask for signals.
      If more than one signal has been received the signal
      with the highest priority will be returned. Priorities are:
      USER1 > USER2 > STOP > TERMINATE. */

  SignalInfo operator()();

  /// send myself a noncatchable stop signal
  static void stopprocess();

private:
  static unsigned int u1; // number of user1 signals received
  static unsigned int u2; // number of user2 signals received
  static unsigned int k; // number of terminate signals received
  static unsigned int s; // number of stop signals received
  static unsigned int count; // total number of signals received
  static bool initialized;

  // functions to be called by the signal handlers, to register them with
  // this object, only for internal use

  static void tstp(int);
  static void kill(int);
  static void usr1(int);
  static void usr2(int);
};

} // end namespace
} // end namespace

#endif // ALPS_SIGNAL_H
