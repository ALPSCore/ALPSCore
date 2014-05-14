/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NGS_STACKTRACE_HPP
#define ALPS_NGS_STACKTRACE_HPP

#include <alps/config.h>
#include <alps/utility/stringify.hpp>

#include <string>

// maximal number of stack frames displayed in stacktrace. Default 63
#ifndef ALPS_NGS_MAX_FRAMES
    #define ALPS_NGS_MAX_FRAMES 63
#endif

// prevent the signal object from registering signals
#ifdef BOOST_MSVC
    #define ALPS_NGS_NO_SIGNALS
#endif

// do not print a stacktrace in error messages
#ifndef __GNUC__
    #define ALPS_NGS_NO_STACKTRACE
#endif

// TODO: have_python
// TODO: have_mpi
// TODO: have_thread



// TODO: check for gcc and use __PRETTY_FUNCTION__

#define ALPS_STACKTRACE (                                                          \
      std::string("\nIn ") + __FILE__                                              \
    + " on " + ALPS_NGS_STRINGIFY(__LINE__)                                        \
    + " in " + __FUNCTION__ + "\n"                                          	   \
    + ::alps::ngs::stacktrace()                                                    \
)

namespace alps {
    namespace ngs {

        ALPS_DECL std::string stacktrace();

    }
}

#endif
