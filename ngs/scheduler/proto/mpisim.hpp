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

#if !defined(ALPS_NGS_SCHEDULER_MPISIM_NG_HPP) && defined(ALPS_HAVE_MPI)
#define ALPS_NGS_SCHEDULER_MPISIM_NG_HPP

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/boost_mpi.hpp>

#include <boost/chrono.hpp>

#include <stdexcept>

namespace alps {

    template<typename Impl> class mpisim_ng : public Impl {

        typedef enum { NOOP_TAG, CHECKPOINT_TAG, FRACTION_TAG, STOP_TAG } tag_type;
        
        public:
    
            using Impl::collect_results;
        
            mpisim_ng(typename alps::parameters_type<Impl>::type const & p) {
                throw std::runtime_error("No communicator passed" + ALPS_STACKTRACE);
            }

            mpisim_ng(typename alps::parameters_type<Impl>::type const & p, boost::mpi::communicator const & c)
                : Impl(p, c.rank())
                , communicator(c)
                , binnumber(p["binnumber"] | std::min(128, 2 * c.size()))
                , fraction(0.)
                , fraction_duration(2.)
                , start_time_point(boost::chrono::high_resolution_clock::now())
                , last_time_point(boost::chrono::high_resolution_clock::now())
                , fraction_time_point(boost::chrono::high_resolution_clock::now())
            {
                MPI_Errhandler_set(communicator, MPI_ERRORS_RETURN);
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
                        boost::mpi::broadcast(communicator, tag, 0);
                        std::string filename_str = filename.string();
                        boost::mpi::broadcast(communicator, filename_str, 0);
                        dynamic_cast<Impl const &>(*this).save(filename_str + "." + cast<std::string>(communicator.rank()));
                    } else
                        const_cast<mpisim_ng<Impl> &>(*this).check_communication();
                }
            }

            bool run(boost::function<bool ()> const & stop_callback) {
                user_stop_callback = stop_callback;
                this->set_status(Impl::running);
                Impl::run(boost::bind<bool>(&mpisim_ng<Impl>::mpi_stop_callback, boost::ref(*this)));
                this->set_status(Impl::finished);
                return stop_callback();
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

            void check_communication() {
                if (this->status() != Impl::interrupted) {
                    boost::chrono::high_resolution_clock::time_point now_time_point = boost::chrono::high_resolution_clock::now();
                    // TODO: make duration a parameter, if running in single thread mpi mode, only check every minute or so ...
                    if (this->status() != Impl::running || now_time_point - last_time_point > boost::chrono::duration<int>(1)) {
                        tag_type tag = NOOP_TAG;
                        if (!communicator.rank()) {
                            if (this->status() == Impl::running && user_stop_callback())
                                tag = STOP_TAG;
                            else if (now_time_point - fraction_time_point > fraction_duration) {
                                tag = FRACTION_TAG;
                                // TODO: add mincheck, maxcheck
                                fraction_duration = boost::chrono::duration<double>(std::min(
                                    2. *  fraction_duration.count(),
                                    // TODO: make interval also smaller ...
                                    std::max(
                                          double(fraction_duration.count())
                                        , 0.8 * (1 - fraction) / fraction * boost::chrono::duration_cast<boost::chrono::duration<double> >(now_time_point - start_time_point).count()
                                    )
                                ));
                                std::cout << "Check complete fraction. Next check in " << fraction_duration.count() << "s" << std::endl;
                                fraction_time_point = boost::chrono::high_resolution_clock::now();
                            }
                        }
                        // TODO: make root a parameter
                        boost::mpi::broadcast(communicator, tag, 0);
                        switch (tag) {
                            case CHECKPOINT_TAG:
                                {
                                    std::string filename;
                                    boost::mpi::broadcast(communicator, filename, 0);
                                    dynamic_cast<Impl &>(*this).save(filename + "." + cast<std::string>(communicator.rank()));
                                }
                                break;
                            case FRACTION_TAG:
                                if (!communicator.rank()) {
                                    double collected;
                                    boost::mpi::reduce(communicator, Impl::fraction_completed(), collected, std::plus<double>(), 0);
                                    fraction = collected;
                                    if (fraction >= 1.) {
                                        tag = STOP_TAG;
                                        // TODO: make root a parameter
                                        boost::mpi::broadcast(communicator, tag, 0);
                                        this->set_status(Impl::interrupted);
                                    }
                                } else
                                    reduce(communicator, Impl::fraction_completed(), std::plus<double>(), 0);
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
        
        protected:
        
            virtual bool work_done() {
                return false;
            }

        private:

            bool mpi_stop_callback() const {
                return this->status() == Impl::interrupted;
            }

            boost::mpi::communicator communicator;
            std::size_t binnumber;
            typename Impl::template atomic<double> fraction;
            boost::chrono::duration<double> fraction_duration;
            boost::chrono::high_resolution_clock::time_point start_time_point;
            boost::chrono::high_resolution_clock::time_point last_time_point;
            boost::chrono::high_resolution_clock::time_point fraction_time_point;
            boost::function<bool ()> user_stop_callback;
    };
}

#endif
