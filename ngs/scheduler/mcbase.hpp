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

#ifndef ALPS_NGS_SCHEDULER_MCBASE_HPP
#define ALPS_NGS_SCHEDULER_MCBASE_HPP

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/config.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/mcresults.hpp> // TODO: replace by new alea
#include <alps/ngs/mcobservables.hpp> // TODO: replace by new alea

#ifdef ALPS_HAVE_PYTHON
    #include <alps/ngs/boost_python.hpp>
#endif

#include <alps/random/mersenne_twister.hpp>

// #include <alps/ngs/config_alea.hpp> TODO: this file does not exits!

#include <boost/chrono.hpp>
#include <boost/function.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/random/variate_generator.hpp>

#include <vector>
#include <string>

namespace alps {

    class ALPS_DECL mcbase {

        public:

            typedef alps::params parameters_type;
            typedef alps::mcresults results_type;
            typedef std::vector<std::string> result_names_type;

            mcbase(parameters_type const & p, std::size_t seed_offset = 0);

            virtual ~mcbase() {}

            virtual void update() = 0;

            virtual void measure() = 0;

            virtual double fraction_completed() const = 0;
        
            void save(boost::filesystem::path const & path) const;

            void load(boost::filesystem::path const & path);

            virtual void save(alps::hdf5::archive & ar) const;

            virtual void load(alps::hdf5::archive & ar);

            bool run(boost::function<bool ()> const & stop_callback);
            
            #ifdef ALPS_HAVE_PYTHON
                bool run(boost::python::object stop_callback);
            #endif

            result_names_type result_names() const;

            result_names_type unsaved_result_names() const;

            results_type collect_results() const;

            virtual results_type collect_results(result_names_type const & names) const;

            // TODO: add function parameters_type & params() { reutrn m_params; } and rename params to m_params
            parameters_type & get_params();

            // TODO: add function parameters_type & measurements() { reutrn m_measurements; } and rename measurements to m_measurements
            mcobservables & get_measurements();
            
            // TODO: add function double random() { reutrn m_random; } and rename random to m_random
            double get_random();

        protected:

            virtual void lock_data();
            virtual void unlock_data();
        
            virtual void lock_results();
            virtual void unlock_results();

            parameters_type params;
            
            //TODO ifdef
            mcobservables measurements;
            
            boost::variate_generator<boost::mt19937, boost::uniform_real<> > random;

        private:

            #ifdef ALPS_HAVE_PYTHON
                static bool callback_wrapper(boost::python::object stop_callback);
            #endif
    };
}

#endif
