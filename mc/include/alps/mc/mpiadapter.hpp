/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <boost/function.hpp>
#include <boost/utility/enable_if.hpp>

#include <alps/config.hpp>

#if defined(ALPS_HAVE_MPI)

#include <alps/accumulators/mpi.hpp>
#include <alps/mc/check_schedule.hpp>

namespace alps {

    namespace detail {

        /// Base class for mcmpiadapter; should never be instantiated by a user

        template<typename Base, typename ScheduleChecker> class mcmpiadapter_base : public Base {

        public:
            typedef typename Base::parameters_type parameters_type;

        protected:
            /// Construct mcmpiadapter_base with a custom scheduler
            mcmpiadapter_base(
                  parameters_type const & parameters
                , alps::mpi::communicator const & comm
                , ScheduleChecker const & check
            )
                : Base(parameters, comm.rank())
                , communicator(comm)
                , schedule_checker(check)
                , clone(comm.rank())
            {}

       public:
            double fraction_completed() const {
                return fraction;
            }

            bool run(boost::function<bool ()> const & stop_callback) {
                bool done = false, stopped = false;
                do {
                    this->update();
                    this->measure();
                    if (stopped || schedule_checker.pending()) {
                        stopped = stop_callback();
                        double local_fraction = stopped ? 1. : Base::fraction_completed();
                        schedule_checker.update(fraction = alps::mpi::all_reduce(communicator, local_fraction, std::plus<double>()));
                        done = fraction >= 1.;
                    }
                } while(!done);
                return !stopped;
            }

            typename Base::results_type collect_results() const {
                return collect_results(this->result_names());
            }

            typename Base::results_type collect_results(typename Base::result_names_type const & names) const {
                typename Base::results_type partial_results;
                for(typename Base::result_names_type::const_iterator it = names.begin(); it != names.end(); ++it) {
                    size_t has_count=(this->measurements[*it].count() > 0);
                    const size_t sum_counts =
                            alps::mpi::all_reduce(communicator,
                                                  has_count,
                                                  std::plus<size_t>());
                    if (static_cast<int>(sum_counts) == communicator.size()) {
                        typename Base::observable_collection_type::value_type merged = this->measurements[*it];
                        merged.collective_merge(communicator, 0);
                        partial_results.insert(*it, merged.result());
                    } else if (sum_counts > 0 && static_cast<int>(sum_counts) < communicator.size()) {
                        throw std::runtime_error(*it + " was measured on only some of the MPI processes.");
                    }
                }
                return partial_results;
            }

        protected:

            alps::mpi::communicator communicator;

            ScheduleChecker schedule_checker;
            double fraction;
            int clone;
        };
    } // detail::


    /// MPI adapter for an MC simulation class, with a general ScheduleChecker
    template<typename Base, typename ScheduleChecker = alps::check_schedule> class mcmpiadapter : public detail::mcmpiadapter_base<Base,ScheduleChecker> {
    private:
        typedef detail::mcmpiadapter_base<Base,ScheduleChecker> base_type_;

    public:
        typedef typename base_type_::parameters_type parameters_type;

        /// Construct mcmpiadapter with a custom scheduler
        // Just forwards to the base class constructor
        mcmpiadapter(
            parameters_type const & parameters
            , alps::mpi::communicator const & comm
            , ScheduleChecker const & check
            )
            : base_type_(parameters, comm, check)
        {}
    };

    /// MPI adapter for an MC simulation class, with default ScheduleChecker
    // partial specialization
    template<typename Base> class mcmpiadapter<Base,alps::check_schedule> : public detail::mcmpiadapter_base<Base,alps::check_schedule> {
    private:
        typedef alps::check_schedule ScheduleChecker;
        typedef detail::mcmpiadapter_base<Base,ScheduleChecker> base_type_;

    public:
        typedef typename base_type_::parameters_type parameters_type;

        /// Construct mcmpiadapter with a custom scheduler
        // Just forwards to the base class constructor
        mcmpiadapter(
            parameters_type const & parameters
            , alps::mpi::communicator const & comm
            , ScheduleChecker const & check
            )
            : base_type_(parameters, comm, check)
        {}

        /// Construct mcmpiadapter_base with alps::check_schedule with the relevant parameters Tmin and Tmax taken from the provided parameters
        // constructs the ScheduleChecker object and then forwards the ctor
        mcmpiadapter(
            parameters_type const & parameters
            , alps::mpi::communicator const & comm
            )
            : base_type_(parameters, comm, ScheduleChecker(parameters["Tmin"], parameters["Tmax"]))
        { }

        /// Define parameters specific for alps::check_schedule: Tmin and Tmax
        static parameters_type& define_parameters(parameters_type & parameters) {
            base_type_::define_parameters(parameters);
            if (parameters.is_restored()) return parameters;
            parameters.template define<std::size_t>("Tmin", 1, "minimum time to check if simulation has finished");
            parameters.template define<std::size_t>("Tmax", 600, "maximum time to check if simulation has finished");
            return parameters;
        }
    };

}

#endif
