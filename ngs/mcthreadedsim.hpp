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

#ifndef ALPS_NGS_MCTHREADEDSIM_HPP
#define ALPS_NGS_MCTHREADEDSIM_HPP

#include <alps/ngs/atomic.hpp>

#ifndef ALPS_NGS_SINGLE_THREAD
    #include <boost/thread.hpp>
#endif

namespace alps {

    #ifndef ALPS_NGS_SINGLE_THREAD

        template<typename Impl> class mcthreadedsim : public Impl {
            public:
                mcthreadedsim(typename parameters_type<Impl>::type const & p, std::size_t seed_offset = 0)
                    : Impl(p, seed_offset)
                {}

                void save(boost::filesystem::path const & path) const {
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    Impl::save(path);
                }

                void load(boost::filesystem::path const & path) {
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    Impl::load(path);
                }

                void do_update() {
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    static_cast<Impl &>(*this).do_update();
                }

                void do_measurements() {
                    boost::lock_guard<boost::mutex> mlock(measurements_mutex);
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    static_cast<Impl &>(*this).do_measurements();
                }

                using Impl::collect_results;

                typename Impl::results_type collect_results(typename Impl::result_names_type const & names) const {
                    boost::lock_guard<boost::mutex> mlock(measurements_mutex);
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    return static_cast<Impl const &>(*this).collect_results();
                }

                double fraction_completed() const {
                    boost::lock_guard<boost::mutex> glock(global_mutex);
                    return static_cast<Impl const &>(*this).fraction_completed();
                }

            private:

                boost::mutex mutable global_mutex;
                boost::mutex mutable measurements_mutex;

        };
    #endif
}

#endif
