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

#if !defined(ALPS_NGS_SCHEDULER_MPIMCSIM_HPP) && defined(ALPS_HAVE_MPI)
#define ALPS_NGS_SCHEDULER_MPIMCSIM_HPP

#include <alps/ngs/stacktrace.hpp>

#include <boost/mpi.hpp>
#include <boost/chrono.hpp>

#include <stdexcept>

namespace alps {

    template<typename Impl> class mcmpisim : public Impl {
        public:
            using Impl::collect_results;
            
            mcmpisim(typename alps::parameters_type<Impl>::type const & p) {
                throw std::runtime_error("No communicator passed" + ALPS_STACKTRACE);
            }

            mcmpisim(typename alps::parameters_type<Impl>::type const & p, boost::mpi::communicator const & c) 
                : Impl(p, c.rank())
                , communicator(c)
                , binnumber(p["binnumber"] | std::min(128, 2 * c.size()))
                , data_locked(false)
                , results_locked(false)
                , fraction(0.)
                , check_duration(8.)
                , start_time_point(boost::chrono::high_resolution_clock::now())
                , last_check_time_point(boost::chrono::high_resolution_clock::now())
            {
                MPI_Errhandler_set(communicator, MPI_ERRORS_RETURN);
            }

            double fraction_completed() const {
                return fraction;
            }

            typename alps::results_type<Impl>::type collect_results(typename alps::result_names_type<Impl>::type const & names) const {
                typename alps::results_type<Impl>::type local_results = Impl::collect_results(names), partial_results;
                for(typename alps::results_type<Impl>::type::iterator it = local_results.begin(); it != local_results.end(); ++it)
                    if (it->second.count())
                        partial_results.insert(it->first, it->second.reduce(communicator, binnumber));
                    else
                        partial_results.insert(it->first, it->second);
                return partial_results;
            }

        protected:

            void lock_data() {
                data_locked = true;
                if (results_locked)
                    check_fraction();
            }
        
            void unlock_data() {
                data_locked = false;
            }
        
            void lock_results() {
                results_locked = true;
                if (data_locked)
                    check_fraction();
            }
        
            void unlock_results() {
                results_locked = false;
            }

        private:

            void check_fraction() {
                boost::chrono::high_resolution_clock::time_point now_time_point = boost::chrono::high_resolution_clock::now();
                if (now_time_point - last_check_time_point > check_duration) {
                    fraction = boost::mpi::all_reduce(communicator, Impl::fraction_completed(), std::plus<double>());
                    check_duration = boost::chrono::duration<double>(std::min(
                        2. *  check_duration.count(),
                        std::max(
                              double(check_duration.count())
                            , 0.8 * (1 - fraction) / fraction * boost::chrono::duration_cast<boost::chrono::duration<double> >(now_time_point - start_time_point).count()
                        )
                    ));
                    last_check_time_point = now_time_point;
                }
            }

            boost::mpi::communicator communicator;
            std::size_t binnumber;
            bool data_locked;
            bool results_locked;
            double fraction;
            boost::chrono::duration<double> check_duration;
            boost::chrono::high_resolution_clock::time_point start_time_point;
            boost::chrono::high_resolution_clock::time_point last_check_time_point;
    };
}

#endif
