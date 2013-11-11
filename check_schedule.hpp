/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2012 by Jan Gukelberger <gukel@comp-phys.org>                     *
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

#ifndef ALPS_NGS_SCHEDULER_CHECK_SCHEDULE_HPP
#define ALPS_NGS_SCHEDULER_CHECK_SCHEDULE_HPP

#include <boost/chrono.hpp>

namespace alps {
    
    class check_schedule
    {
    public:
        typedef boost::chrono::high_resolution_clock clock;
        typedef clock::time_point time_point;
        typedef boost::chrono::duration<double> duration;

        check_schedule(double tmin=60., double tmax=900.)
        :   min_check_(tmin)
        ,   max_check_(tmax)
        {
        }

        bool pending() const
        {
            time_point now = clock::now();
            return now > (last_check_time_ + next_check_);
        }
    
        double check_interval() const
        {
            return next_check_.count();
        }

        void update(double fraction)
        {
            time_point now = clock::now();

            // first check
            if( start_time_ == time_point() )
            {
                start_time_ = now;
                start_fraction_ = fraction;
            }

            if( fraction > start_fraction_ )
            {
                // estimate remaining time; propose to run 1/4 of that time
                duration old_check = next_check_;
                next_check_ = 0.25 * (1 - fraction) * (now - start_time_) / (fraction - start_fraction_);
                if( next_check_ > 2*old_check ) next_check_ = 2 * old_check;
                if( next_check_ < min_check_ ) next_check_ = min_check_;
                if( next_check_ > max_check_ ) next_check_ = max_check_;
            }
            else
                next_check_ = min_check_;

            last_check_time_ = now;
        }

    private:
        duration min_check_;
        duration max_check_;

        time_point start_time_;
        time_point last_check_time_;
        double start_fraction_;
        duration next_check_;
    
    };

} // namespace alps 

#endif // !defined ALPS_NGS_SCHEDULER_CHECK_SCHEDULE_HPP
