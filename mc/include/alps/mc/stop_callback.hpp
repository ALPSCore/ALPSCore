/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_CALLBACK_HPP
#define ALPS_CALLBACK_HPP

#include <alps/config.hpp>
#include <alps/utilities/signal.hpp>

#include <alps/mc/check_schedule.hpp>

#ifdef ALPS_HAVE_MPI
#include <boost/optional.hpp>
# include "alps/utilities/mpi.hpp"
#endif

namespace alps {
        /// Functor-predicate:  is it time to stop?
        /** The functor is initialized with desired duration, and returns `true`
            when the root process receives a signal or time runs out on the root.
        */
	class stop_callback {
                private:
                    typedef detail::posix_wall_clock clock_type;
                    typedef clock_type::time_duration_type time_duration_type;
                    typedef clock_type::time_point_type time_point_type;
                
		public:
                    /// Initializes the functor with the desired time duration
                    /** @param timelimit Time limit (seconds); 0 means "indefinitely" */
		    stop_callback(std::size_t timelimit);
#ifdef ALPS_HAVE_MPI
                    /// Initializes the functor with the desired time duration
                    /** @param timelimit Time limit (seconds); 0 means "indefinitely"
                        @param cm MPI communicator to determine the root process
                     */
                    stop_callback(alps::mpi::communicator const & cm, std::size_t timelimit);
#endif
                    /// Returns `true` if it's time to stop (time is up or signal is received)
		    bool operator()() const;
		private:
                    const time_duration_type limit;
		    alps::signal signals;
		    const time_point_type start;
#ifdef ALPS_HAVE_MPI
                    boost::optional<alps::mpi::communicator> comm;
#endif
	};

        /// Functor-predicate:  is it time to stop?
        /** The functor is initialized with desired duration, and returns `true`
            when times runs out .
        */
	class simple_time_callback {
                private:
                    typedef detail::posix_wall_clock clock_type;
                    typedef clock_type::time_duration_type time_duration_type;
                    typedef clock_type::time_point_type time_point_type;
                
		public:
                    /// Initializes the functor with the desired time duration
                    /** @param timelimit Time limit (seconds); 0 means "indefinitely" */
		    simple_time_callback(std::size_t timelimit);

                    /// Returns `true` if time is up
		    bool operator()() const;
		private:
		    const time_duration_type limit;
		    const time_point_type start;
	};
}

#endif
