/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/signal.hpp>
#include <alps/utilities/stacktrace.hpp>
//#include <alps/hdf5/archive.hpp> //FIXME - see below

#include <cstring>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <cstdio>

#include <signal.h>

namespace alps {

    signal::signal() {
        #if not ( defined BOOST_MSVC || defined ALPS_NO_SIGNALS )
            static bool initialized;
            if (!initialized) {
                initialized = true;

                static struct sigaction action;
                memset(&action, 0, sizeof(action));
                action.sa_handler = &signal::slot;
                sigaction(SIGINT, &action, NULL);
                sigaction(SIGTERM, &action, NULL);
                sigaction(SIGXCPU, &action, NULL);
                sigaction(SIGQUIT, &action, NULL);
                sigaction(SIGUSR1, &action, NULL);
                sigaction(SIGUSR2, &action, NULL);
            }
        #endif
        listen();
    }

    bool signal::empty() const {
        return end_  == begin_;
    }

    int signal::top() const {
        return signals_[(end_ - 1) & 0x1F];
    }

    void signal::pop() {
        --end_ &= 0x1F;
    }

    void signal::listen() {
        #if not ( defined BOOST_MSVC || defined ALPS_NO_SIGNALS )
            static bool initialized;
            if (!initialized) {
                initialized = true;

                static struct sigaction action;
                memset(&action, 0, sizeof(action));
                action.sa_handler = &signal::segfault;
                sigaction(SIGSEGV, &action, NULL);
                sigaction(SIGBUS, &action, NULL);
            }
        #endif
    }

    void signal::slot(int signal) {
        fprintf(stderr, "Received signal %i\n", signal);
        signals_[end_] = signal;
        ++end_ &= 0x1F;
        if (begin_ == end_)
            ++begin_ &= 0x1F;
    }

    void signal::segfault(int signal) {
        //hdf5::archive::abort(); //FIXME
        fprintf(stderr, "Abort by signal %i\n", signal);
        std::cerr << ALPS_STACKTRACE;
        std::abort();
    }

    std::size_t signal::begin_ = 0;
    std::size_t signal::end_ = 0;
    boost::array<int, 32> signal::signals_;
}
