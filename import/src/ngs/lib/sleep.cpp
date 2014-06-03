/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
