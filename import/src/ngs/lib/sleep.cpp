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

#include <alps/ngs/sleep.hpp>

#ifndef ALPS_NGS_SINGLE_THREAD

#include <boost/thread.hpp>
#include <boost/thread/xtime.hpp>

namespace alps {

    void sleep(std::size_t nanoseconds) {
        // TODO: check if boost::this_thread::sleep is nicer than xtime
        boost::xtime xt;
#if BOOST_VERSION < 105000
        boost::xtime_get(&xt, boost::TIME_UTC);
#else
        boost::xtime_get(&xt, boost::TIME_UTC_);
#endif
        xt.nsec += nanoseconds;
        boost::thread::sleep(xt);
    }
}

#else

#include <ctime>
#include <stdexcept>

namespace alps {

    void sleep(std::size_t nanoseconds) {

        struct timespec tim, tim2;
        tim.tv_nsec = nanoseconds;

        if(nanosleep(&tim , &tim2) < 0)
            throw std::runtime_error("Nano sleep failed");
    }
}

#endif
