/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <ctime>

namespace alps {

    namespace detail {
        /// Class template to check for simulation completion with adaptive check frequency
        /** Typename CLOCK_T provides time-related types and methods */
        // (Note: We could use traits, but we don't)
        template <typename CLOCK_T>
        class generic_check_schedule {
            public:
            typedef CLOCK_T clock_type;
            typedef typename clock_type::time_point_type time_point_type;
            typedef typename clock_type::time_duration_type time_duration_type;

            private:
            clock_type clock_;
            
            bool next_check_known_;
            double start_fraction_;

            time_duration_type min_check_;
            time_duration_type max_check_;

            time_point_type start_time_;
            time_point_type last_check_time_;

            time_duration_type next_check_;

            public:
            /// Constructor using default clock instance
            /**
               \param[in] tmin minimum time to check if simulation has finished 
               \param[in] tmax maximum time to check if simulation has finished
            */
            generic_check_schedule(double tmin, double tmax)
                : clock_(),
                  next_check_known_(false),
                  start_fraction_(),
                  min_check_(tmin),
                  max_check_(tmax),
                  start_time_(),
                  last_check_time_(),
                  next_check_()
            { }

            /// Constructor using a given clock instance
            /**
               \param[in] tmin minimum time to check if simulation has finished 
               \param[in] tmax maximum time to check if simulation has finished
               \param[in] clock Clock object to use
            */
            generic_check_schedule(double tmin, double tmax, const clock_type& clock)
                : clock_(clock),
                  next_check_known_(false),
                  min_check_(tmin),
                  max_check_(tmax),
                  start_time_(),
                  last_check_time_(),
                  next_check_()
            { }

            /// Returns `true` if it's time to check progress and to schedule new check
            bool pending() const
            {
                if (!next_check_known_) return true;
                time_point_type now = clock_.now_time();
                return clock_.time_diff(now, last_check_time_) > next_check_;
            }
    
            /// Schedule the next check based on the fraction of the simulation completed
            void update(double fraction)
            {
                time_point_type now = clock_.now_time();
                // first check

                if (!next_check_known_)
                {
                    start_time_ = now;
                    start_fraction_ = fraction;
                    next_check_known_=true;
                }

                if (fraction > start_fraction_)
                {
                    // estimate remaining time; propose to run 1/4 of that time
                    time_duration_type old_check = next_check_;
                    next_check_ = 0.25 * (1 - fraction)
                                  * clock_.time_diff(now, start_time_) / (fraction - start_fraction_);
                    
                    if( next_check_ > 2*old_check ) next_check_ = 2 * old_check;
                    if( next_check_ < min_check_ ) next_check_ = min_check_;
                    if( next_check_ > max_check_ ) next_check_ = max_check_;
                }
                else
                    next_check_ = min_check_;

                last_check_time_ = now;
            }
        };


        /// Type for POSIX wall-clock time
        class posix_wall_clock {
          public:
            /// Type for "point at time" (that is, duration from some epoch)
            typedef std::time_t time_point_type;

            /// Type for "duration of time" (that is, difference between to points in time)
            typedef double time_duration_type;

            /// Returns current time point
            static time_point_type now_time() { return std::time(0); }

            /// Returns a difference (duration) between time points
            static time_duration_type time_diff(time_point_type t1, time_point_type t0)
            {
                return std::difftime(t1, t0);
            }
        };
    } // detail::
        
    typedef detail::generic_check_schedule<detail::posix_wall_clock> check_schedule;

} // namespace alps 
