/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#if defined(ALPS_HAVE_MPI)

#include <alps/utility/boost_mpi.hpp>
#include <alps/mc/check_schedule.hpp>

namespace alps {

    template<typename Base, typename ScheduleChecker = alps::check_schedule> class mcmpiadapter : public Base {

        public:

            mcmpiadapter(
                  typename Base::parameters_type const & parameters
                , boost::mpi::communicator const & comm
                , ScheduleChecker const & check = ScheduleChecker()
            )
                : Base(parameters, comm.rank())
                , communicator(comm)
                , schedule_checker(check)
                , clone(comm.rank())
            {}

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
                        schedule_checker.update(fraction = boost::mpi::all_reduce(communicator, local_fraction, std::plus<double>()));
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
                    if (communicator.rank() == 0) {
                        if (this->measurements[*it].count()) {
                            typename Base::observable_collection_type::value_type merged = this->measurements[*it];
                            merged.collective_merge(communicator, 0);
                            partial_results.insert(*it, merged.result());
                        } else
                            partial_results.insert(*it, this->measurements[*it].result());
                    } else if (this->measurements[*it].count())
                        this->measurements[*it].collective_merge(communicator, 0);
                }
                return partial_results;
            }

        protected:

            boost::mpi::communicator communicator;

            ScheduleChecker schedule_checker;
            double fraction;
            int clone;
    };
}

#endif
