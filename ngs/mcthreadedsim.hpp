/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                           Matthias Troyer <troyer@comp-phys.org>                *
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

#ifndef ALPS_NGS_MCTHREADEDSIM_HPP
#define ALPS_NGS_MCTHREADEDSIM_HPP

#ifndef ALPS_NGS_SINGLE_THREAD
    #include <boost/thread.hpp>
#endif

namespace alps {

    #ifndef ALPS_NGS_SINGLE_THREAD
        template<typename T> class mcatomic {
            public:
                mcatomic() {}
                mcatomic(T const & v): value(v) {}
                mcatomic(mcatomic const & v): value(v.value) {}

                mcatomic & operator=(mcatomic const & v) {
                    boost::lock_guard<boost::mutex> lock(mutex);
                    value = v;
                }

                operator T() const { 
                    boost::lock_guard<boost::mutex> lock(mutex);
                    return value; 
                }
            private:
                T volatile value;
                boost::mutex mutable mutex;
        };

        template<typename Impl> class mcthreadedsim : public Impl {
            public:
                mcthreadedsim(typename parameters_type<Impl>::type const & p)
                    : Impl(p)
                    , stop_flag(false)
                {}

                bool run(boost::function<bool ()> const & stop_callback) {
                    boost::thread thread(boost::bind<bool>(&Impl::run, static_cast<Impl *>(this), &mcthreadedsim<Impl>::dummy_callback));
                    checker(stop_callback);
                    thread.join();
                    return !stop_callback();
                }

            protected:

                virtual bool complete_callback(boost::function<bool ()> const &) {
                    return stop_flag;
                }

                virtual void checker(boost::function<bool ()> const & stop_callback) {
                    while (true) {
                        usleep(0.1 * 1e6);
                        if (stop_flag = Impl::complete_callback(stop_callback))
                            return;
                    }
                }

                mcatomic<bool> stop_flag;
                // boost::mutex mutex;
                // measurements and configuration need to be locked separately

            private:

                static bool dummy_callback() {}
        };
    #endif
}

#endif
