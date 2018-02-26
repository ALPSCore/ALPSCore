/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_SIGNAL_HPP
#define ALPS_SIGNAL_HPP

#include <alps/utilities/stacktrace.hpp>

#include <boost/array.hpp>

#include <vector>

namespace alps {

    class signal{

    public:

      /*!
Listen to the following posix signals SIGINT, SIGTERM, SIGXCPU, SIGQUIT, SIGUSR1, SIGUSR2, SIGSTOP SIGKILL. Those signals can be check by empty, top, pop

\verbatim embed:rst
.. note::
   If a SIGSEGV (segfault) or SIGBUS (bus error) occures, a stacktrace
   is printed an all open hdf5 archives are closed before it exits.
\endverbatim
      */
      signal();

      /*!
Returns if a signal has been captured.
      */
      bool empty() const;

      /*!
Returns the last signal that has been captured .
      */
      int top() const;


      /*!
Pops a signal form the stack.
       */
      void pop();

      /*!
Listen to the signals SIGSEGV (segfault) and SIGBUS (bus error). If one
of these signals are captured, a stacktrace is printed an all open hdf5
archives are closed before it exits. 
      */
      static void listen();

      static void slot(int signal);

      static void segfault(int signal);

    private:

      static std::size_t begin_;
      static std::size_t end_;
      static boost::array<int, 0x20> signals_;
    };
}

#endif
