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

#ifndef ALPS_NGS_MCBASE_HPP
#define ALPS_NGS_MCBASE_HPP

#include <alps/ngs/params.hpp>
#include <alps/ngs/mcresults.hpp>
#include <alps/ngs/mcobservables.hpp>

#include <alps/config.h>

#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/random/uniform_real.hpp>
#include <alps/random/mersenne_twister.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/date_time/gregorian/gregorian.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>

#include <vector>
#include <string>

namespace alps {
    class mcbase {
        public:
            typedef alps::params parameters_type;
            typedef mcresults results_type;
            typedef std::vector<std::string> result_names_type;

            mcbase(parameters_type const & p, std::size_t seed_offset = 0)
                : params(p)
                  // TODO: this ist not the best solution - any idea?
                , random(boost::mt19937(static_cast<std::size_t>(p.value_or_default("SEED", 42)) + seed_offset), boost::uniform_real<>())
                , fraction(0.)
                , next_check(8)
                , start_time(boost::posix_time::second_clock::local_time())
                , check_time(boost::posix_time::second_clock::local_time() + boost::posix_time::seconds(next_check))
            {}

            virtual ~mcbase() {}

            virtual void do_update() = 0;

            virtual void do_measurements() = 0;

            virtual double fraction_completed() const = 0;

            void save(boost::filesystem::path const & path) const;

            void load(boost::filesystem::path const & path);

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
            std::size_t next_check;
            boost::posix_time::ptime start_time;
            boost::posix_time::ptime check_time;
    };
}

#endif
