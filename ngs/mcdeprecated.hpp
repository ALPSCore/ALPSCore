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

#ifndef ALPS_NGS_MCDEPRECATED_HPP
#define ALPS_NGS_MCDEPRECATED_HPP

#include <alps/ngs/mcbase.hpp>

#include <alps/config.h>
#include <alps/parameter.h>
#include <alps/alea/observableset.h>

namespace alps {

    class mcdeprecated : public mcbase {

        public:

            mcdeprecated(parameters_type const & p, std::size_t seed_offset = 0);

            double fraction_completed() const;

            virtual double work_done() const = 0;

            virtual void dostep() = 0;

            double random_real(double a = 0., double b = 1.);

            virtual void do_update();

            virtual void do_measurements();

        protected:

            Parameters parms;
// TODO: fix this!
//            ObservableSet & measurements;
            ObservableSet measurements;
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > & random_01;

        private:

            static Parameters make_alps_parameters(parameters_type const & s);

    };

}

#endif
