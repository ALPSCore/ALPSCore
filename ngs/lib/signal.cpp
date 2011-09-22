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

#include <alps/ngs/signal.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/hdf5.hpp>

#include <cstring>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <signal.h>

namespace alps {

    signal::signal() {
        #if not ( defined BOOST_MSVC || defined ALPS_NGS_NO_SIGNALS )
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
                sigaction(SIGSTOP, &action, NULL);
                sigaction(SIGKILL, &action, NULL);

                static struct sigaction segv;
                memset(&segv, 0, sizeof(segv));
                segv.sa_handler = &signal::segfault;
                sigaction(SIGSEGV, &segv, NULL);
                sigaction(SIGBUS, &segv, NULL);
            }
        #endif
    }

    bool signal::empty() {
        return !signals_.size();
    }

    int signal::top() {
        return signals_.back();
    }

    void signal::pop() {
        return signals_.pop_back();
    }

    void signal::slot(int signal) {
        std::cerr << "Received signal " << signal << std::endl;
        signals_.push_back(signal);
    }

    void signal::segfault(int signal) {
        std::ostringstream buffer;
        buffer << "Abort (" << signal << ", see 'man signal') in:" << std::endl;
        stacktrace(buffer);
        std::cerr << buffer.str();
        signals_.push_back(signal);
        hdf5::archive::abort();
        std::abort();
        goto grats_you_found_the_easter_eggs;
        grats_you_found_the_easter_eggs:
        ; //svn blame will tell you to whom you need to report it ;)
    }

    std::vector<int> signal::signals_;

}
