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

#ifndef ALPS_NGS_MULTITHREADEDSIM_HPP
#define ALPS_NGS_MULTITHREADEDSIM_HPP

#include <alps/ngs/api.hpp>
#ifndef ALPS_NGS_SINGLE_THREAD

    #include <alps/ngs/atomic.hpp>
    #include <boost/thread.hpp>

#endif
/*
namespace alps {

    #ifndef ALPS_NGS_SINGLE_THREAD

        template<typename Impl> class multithread : public Impl {
            public:
                multithread(typename alps::parameters_type<Impl>::type const & p, std::size_t seed_offset = 0)
                    : Impl(p, seed_offset)
                {}

                using Impl::save;
                using Impl::load;

                void save(alps::hdf5::archive & ar) const {
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    Impl::save(ar);
                }

                void load(alps::hdf5::archive & ar) {
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    Impl::load(ar);
                }

                using Impl::collect_results;

                typename Impl::results_type collect_results(typename Impl::result_names_type const & names) const {
                    boost::lock_guard<boost::mutex> mlock(measurements_mutex);
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    return Impl::collect_results(names);
                }



            protected:

                void lock_data() {
                    data_guard.reset(new boost::lock_guard<boost::mutex> data_lock(data_mutex));
                }

                void unlock_data() {
                    data_guard.reset();
                }
            
                void lock_results() {
                    data_guard.reset(new boost::lock_guard<boost::mutex> result_guard(result_mutex));
                }
            
                void unlock_results() {
                    result_guard.reset();
                }

            private:

                boost::mutex mutable data_mutex;
                boost::mutex mutable result_mutex;

                scoped_ptr<boost::lock_guard<boost::mutex> > data_guard;
                scoped_ptr<boost::lock_guard<boost::mutex> > result_guard;
        };
    #endif
}
*/
#endif
