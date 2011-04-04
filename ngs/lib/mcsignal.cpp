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

#include <alps/ngs/mcsignal.hpp>

#include <cstring>
#include <iostream>
#include <signal.h>

namespace alps {

    mcsignal::mcsignal() {
        #ifndef BOOST_MSVC
            static bool initialized;
            if (!initialized) {
                static struct sigaction action;
                initialized = true;
                memset(&action, 0, sizeof(action));
                action.sa_handler = &mcsignal::slot;
                sigaction(SIGINT, &action, NULL);
                sigaction(SIGTERM, &action, NULL);
                sigaction(SIGXCPU, &action, NULL);
                sigaction(SIGQUIT, &action, NULL);
                sigaction(SIGUSR1, &action, NULL);
                sigaction(SIGUSR2, &action, NULL);
                sigaction(SIGSTOP, &action, NULL);
            }
        #endif
    }

    bool mcsignal::empty() {
        return !signals_.size();
    }

    int mcsignal::top() {
        return signals_.back();
    }

    void mcsignal::pop() {
        return signals_.pop_back();
    }

    void mcsignal::slot(int signal) {
        std::cerr << "Received signal " << signal << std::endl;
        signals_.push_back(signal);
    }

    std::vector<int> mcsignal::signals_;

}
