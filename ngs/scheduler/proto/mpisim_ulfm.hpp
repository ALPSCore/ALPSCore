/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2013 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Donjan Rdoic <drodic@phys.ethz.ch>                 *
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

/*
 * For a schematic overview on and around check_communication(), refer to:
 * The bottom of this file.
 */

#if !defined(ALPS_NGS_SCHEDULER_MPISIM_NG_HPP) && defined(ALPS_HAVE_MPI)
#define ALPS_NGS_SCHEDULER_MPISIM_NG_HPP

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/boost_mpi.hpp>
#include <alps/ngs/ulfm.hpp>

#include <boost/chrono.hpp>

#include <iomanip>
#include <iostream>
#include <stdexcept>

namespace alps {

    template<typename Impl> class mpisim_ng : public Impl {

        typedef enum { NOOP_TAG, CHECKPOINT_TAG, FRACTION_TAG, STOP_TAG } tag_type;
        
        public:
    
            using Impl::collect_results;
        
            mpisim_ng(typename alps::parameters_type<Impl>::type const & p) {
                throw std::runtime_error("No communicator passed" + ALPS_STACKTRACE);
            }

            mpisim_ng(typename alps::parameters_type<Impl>::type const & p, boost::mpi::communicator & c, double Tmin = 1, double Tmax = 600)
                : Impl(p, c.rank())
                , communicator(c)
                , binnumber(p["binnumber"] | std::min(128, 2 * c.size()))
                , rank(c.rank())
                , fraction(0.)
                , min_check(Tmin)
                , max_check(Tmax)
                , next_check(Tmin)
                , fraction_duration(2.)
                , start_time_point(boost::chrono::high_resolution_clock::now())
                , last_time_point(boost::chrono::high_resolution_clock::now())
                , fraction_time_point(boost::chrono::high_resolution_clock::now())
            {
                MPI_Comm_set_errhandler(communicator, MPI_ERRORS_RETURN);
            }

            double fraction_completed() const {
                return fraction;
            }

            void save(boost::filesystem::path const & filename) const {
                if (this->status() != Impl::interrupted) {
                    if (!communicator.rank()) {
                        tag_type tag = CHECKPOINT_TAG;
                        // TODO: make root a parameter
                        // TODO: use iBcast ...
                        ULFM_CHECK_RANK_FAILURE_BEGIN
                            boost::mpi::broadcast(communicator, tag, 0);
                        ULFM_CHECK_RANK_FAILURE_END(communicator)
                        std::string filename_str = filename.string();
                        ULFM_CHECK_RANK_FAILURE_BEGIN
                            boost::mpi::broadcast(communicator, filename_str, 0);
                        ULFM_CHECK_RANK_FAILURE_END(communicator)
                        dynamic_cast<Impl const &>(*this).save(filename_str);
                    } else
                        const_cast<mpisim_ng<Impl> &>(*this).check_communication();
                }
            }
        
            using Impl::run;

            bool run(
                  boost::function<bool ()> const & stop_callback
                , boost::function<void (double)> const & progress_callback
            ) {
                user_stop_callback = stop_callback;
                this->set_status(Impl::running);
                Impl::run(boost::bind<bool>(&mpisim_ng<Impl>::mpi_stop_callback, boost::ref(*this)), progress_callback);
                this->set_status(Impl::finished);
                return stop_callback();
            }

            typename alps::results_type<Impl>::type collect_results(typename alps::result_names_type<Impl>::type const & names) const {
                typename alps::results_type<Impl>::type local_results = Impl::collect_results(names), partial_results;
                for(typename alps::results_type<Impl>::type::iterator it = local_results.begin(); it != local_results.end(); ++it)
                    if (it->second.count()) {
                        ULFM_CHECK_RANK_FAILURE_BEGIN
                            partial_results.insert(it->first, it->second.reduce(communicator, binnumber));
                        ULFM_CHECK_RANK_FAILURE_END(communicator)
                    } else {
                        ULFM_CHECK_RANK_FAILURE_BEGIN
                            partial_results.insert(it->first, it->second);
                        ULFM_CHECK_RANK_FAILURE_END(communicator)
                    }
                return partial_results;
            }

            void check_communication() {
                if (this->status() != Impl::interrupted) {
                    boost::chrono::high_resolution_clock::time_point now_time_point = boost::chrono::high_resolution_clock::now();
                    // TODO: make duration a parameter, if running in single thread mpi mode, only check every minute or so ...
                    // TODO: measure how long a communication takes and make checking adaptive ... (always on dualthread, every minute on singlethread)
                    if (this->status() != Impl::running || now_time_point - last_time_point > boost::chrono::duration<double>(next_check) || user_stop_callback()) {
                        tag_type tag = NOOP_TAG;
                        if (this->status() == Impl::running && !communicator.rank())
                            tag = user_stop_callback() ?  STOP_TAG : FRACTION_TAG;
                        // TODO: make root a parameter
                        ULFM_CHECK_RANK_FAILURE_BEGIN
                            boost::mpi::broadcast(communicator, tag, 0);
                        ULFM_CHECK_RANK_FAILURE_END(communicator)
                        switch (tag) {
                            case CHECKPOINT_TAG:
                                {
                                    std::string filename;
                                    ULFM_CHECK_RANK_FAILURE_BEGIN
                                        boost::mpi::broadcast(communicator, filename, 0);
                                    ULFM_CHECK_RANK_FAILURE_END(communicator)
                                    dynamic_cast<Impl &>(*this).save(filename);
                                }
                                break;
                            case FRACTION_TAG:
                                if (!communicator.rank()) {
                                    double collected;
                                    ULFM_CHECK_RANK_FAILURE_BEGIN
                                        boost::mpi::reduce(communicator, Impl::fraction_completed(), collected, std::plus<double>(), 0);
                                    ULFM_CHECK_RANK_FAILURE_END(communicator)
                                    fraction = collected;
                                    if (fraction >= 1.) {                                        
                                        ULFM_IGNORE_RANK_FAILURE_BEGIN
                                            boost::mpi::broadcast(communicator, next_check = 0, 0);
                                        ULFM_IGNORE_RANK_FAILURE_END
                                        tag = STOP_TAG;
                                        // TODO: make root a parameter
                                        ULFM_CHECK_RANK_FAILURE_BEGIN
                                            boost::mpi::broadcast(communicator, tag, 0);
                                        ULFM_CHECK_RANK_FAILURE_END(communicator)

                                        this->set_status(Impl::interrupted);
                                        break;
                                    } else {
                                        double elapsed = boost::chrono::duration_cast<boost::chrono::duration<double> >(now_time_point - start_time_point).count();
                                        // TODO: save first fraction (if loaded from checkpoint ...)
                                        double start_fraction = 0;
                                        next_check = std::max(
                                              min_check
                                            , std::min(
                                                  max_check
                                                , std::min(
                                                      2 * next_check
                                                    , std::max(
                                                          next_check/ 2
                                                        , elapsed / 4. * (1 - fraction) / (fraction - start_fraction)
                                                      )
                                                  )
                                              )
                                          );
                                        std::cout << std::setprecision(2) << 100 * collected << "% done, next check in " << static_cast<std::size_t>(next_check) << "s" << std::endl;
                                    }
                                } else {
                                    ULFM_CHECK_RANK_FAILURE_BEGIN
                                        reduce(communicator, Impl::fraction_completed(), std::plus<double>(), 0);
                                    ULFM_CHECK_RANK_FAILURE_END(communicator)
                                }
                                ULFM_IGNORE_RANK_FAILURE_BEGIN
                                    boost::mpi::broadcast(communicator, next_check, 0);
                                ULFM_IGNORE_RANK_FAILURE_END
                                break;
                            case STOP_TAG:
                                this->set_status(Impl::interrupted);
                                break;
                            case NOOP_TAG:
                                break;
                        }
                        last_time_point = boost::chrono::high_resolution_clock::now();
                    }
                }
            }

            std::string file_suffix() const {
                return "." + cast<std::string>(rank);
            }

        protected:
        
            virtual bool work_done() {
                return false;
            }

        private:

            bool mpi_stop_callback() const {
                return this->status() == Impl::interrupted;
            }

            boost::mpi::communicator & communicator;
            std::size_t binnumber;
            int rank;
            typename Impl::template atomic<double> fraction;
            double min_check;
            double max_check;
            double next_check;
            boost::chrono::duration<double> fraction_duration;
            boost::chrono::high_resolution_clock::time_point start_time_point;
            boost::chrono::high_resolution_clock::time_point last_time_point;
            boost::chrono::high_resolution_clock::time_point fraction_time_point;
            boost::function<bool ()> user_stop_callback;
    };
}

#endif

/*
Communication flow for save() and check_communication().
Same capital letters correspond to matching collectives: collAi matches collAj
pass/barrier refer to the usage of ULFM ignore/check routines.
0: is the root node
k: are all other ranks

save:
    0:  bcastA3, barrier
        bcastB2, barrier
    k:  check_communication

check_communication:
    if time to check:
        bcastA1, barrier
        CHECKPOINT:
            bcastB1, barrier
        FRACTION:
            0:  reduceC1, barrier
                frac>=1:
                    bcastD2, pass
                    bcastA2, barrier
                    break
                else:
                    - (print %done)
            k:  reduceC2, barrier
            bcastD1, pass
        STOP:
            -
        NOOP:
            -
*/
