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

#ifndef ALPS_NGS_BASE_HPP
#define ALPS_NGS_BASE_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/mcresults.hpp>
#include <alps/ngs/mcobservables.hpp>

#include <alps/random/mersenne_twister.hpp>

#include <boost/chrono.hpp>
#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <vector>
#include <string>

namespace alps {

    class ALPS_DECL base {

        public:

            typedef alps::params parameters_type;
            typedef alps::mcresults results_type;
            typedef std::vector<std::string> result_names_type;

            base(parameters_type const & p, std::size_t seed_offset = 0)
                : params(p)
                  // TODO: this ist not the best solution - any idea?
                , random(boost::mt19937(p["SEED"].or_default(42) + seed_offset), boost::uniform_real<>())
                , fraction(0.)
                , check_duration(8.)
                , start_time_point(boost::chrono::high_resolution_clock::now())
                , last_check_time_point(boost::chrono::high_resolution_clock::now())
            {}

            virtual ~base() {}

            virtual void do_update() = 0;

            virtual void do_measurements() = 0;

            virtual double fraction_completed() const = 0;

			// TODO: add boost::filesystem version
            void save(std::string const & filename) const;

			// TODO: add boost::filesystem version
            void load(std::string const & filename);

            virtual void save(alps::hdf5::archive & ar) const;

            virtual void load(alps::hdf5::archive & ar);

            bool run(boost::function<bool ()> const & stop_callback);

            result_names_type result_names() const;

            result_names_type unsaved_result_names() const;

            results_type collect_results() const;

            virtual results_type collect_results(result_names_type const & names) const;

        protected:

            virtual bool complete_callback(boost::function<bool ()> const & stop_callback);

            parameters_type params;
            mcobservables measurements;
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > random;

        private:

            double fraction;
            boost::chrono::duration<double> check_duration;
            boost::chrono::high_resolution_clock::time_point start_time_point;
            boost::chrono::high_resolution_clock::time_point last_check_time_point;

    };
}

#endif
