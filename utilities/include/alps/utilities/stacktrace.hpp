/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_STACKTRACE_HPP
#define ALPS_STACKTRACE_HPP

#include <alps/config.hpp>
#include <alps/utilities/stringify.hpp>

#include <string>

// maximal number of stack frames displayed in stacktrace. Default 63
#ifndef ALPS_MAX_FRAMES
    #define ALPS_MAX_FRAMES 63
#endif

// prevent the signal object from registering signals
#ifdef BOOST_MSVC
    #define ALPS_NO_SIGNALS
#endif

// do not print a stacktrace in error messages
#ifndef __GNUC__
    #define ALPS_NO_STACKTRACE
#endif

// TODO: have_mpi
// TODO: have_thread



// TODO: check for gcc and use __PRETTY_FUNCTION__

#define ALPS_STACKTRACE (                                                          \
      std::string("\nIn ") + __FILE__                                              \
    + " on " + ALPS_STRINGIFY(__LINE__)                                            \
    + " in " + __FUNCTION__ + "\n"                                          	   \
    + ::alps::stacktrace()                                                    \
)

namespace alps {

    std::string stacktrace();

}

#endif
