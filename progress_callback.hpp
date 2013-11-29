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

#ifndef ALPS_NGS_SCHEDULER_PROGRESS_CALLBACK_HPP
#define ALPS_NGS_SCHEDULER_PROGRESS_CALLBACK_HPP

#include <alps/check_schedule.hpp>
#include <iostream>

namespace alps {
    
    class ALPS_DECL progress_callback
    {
    public:
        progress_callback(double tmin=60., double tmax=900.) : schedule_(tmin,tmax) {}
    
        /// print current progress fraction to cout if schedule says we should
        void operator()(double fraction)
        {
            if( schedule_.pending() )
            {
                schedule_.update(fraction);

                std::streamsize oldprec = std::cout.precision(3);
                std::cout << "Completed " << 100*fraction << "%.";
                if( fraction < 1. ) 
                    std::cout << " Next check in " << schedule_.check_interval() << " seconds." << std::endl;
                else
                    std::cout << " Done." << std::endl;
                std::cout.precision(oldprec);
            }
        }
    
    private:
        check_schedule schedule_;
    };

}

#endif // !defined ALPS_NGS_SCHEDULER_PROGRESS_CALLBACK_HPP
