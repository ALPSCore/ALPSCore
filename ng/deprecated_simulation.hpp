/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2010 by Lukas Gamper <gamperl@gmail.com>
 *                       Matthias Troyer <troyer@comp-phys.org>
 *
 * This software is part of the ALPS libraries, published under the ALPS
 * Library License; you can use, redistribute it and/or modify it under
 * the terms of the license, either version 1 or (at your option) any later
 * version.
 * 
 * You should have received a copy of the ALPS Library License along with
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
 * available from http://alps.comp-phys.org/.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
 * DEALINGS IN THE SOFTWARE.
 *
 *****************************************************************************/

#include <alps/hdf5.hpp>
#include <alps/ng/api.hpp>
#include <alps/ng/alea.hpp>
#include <alps/parameter.h>
#include <alps/ng/boost.hpp>
#include <alps/ng/signal.hpp>
#include <alps/ng/parameters.hpp>
#include <alps/ng/single_simulation.hpp>

#include <boost/mpi.hpp>
#include <boost/bind.hpp>
#include <boost/utility.hpp>
#include <boost/variant.hpp>
#include <boost/function.hpp>
#include <boost/mpl/vector.hpp>
#include <boost/mpl/contains.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/program_options.hpp>
#include <boost/random/uniform_real.hpp>
#include <boost/ptr_container/ptr_map.hpp>
#include <boost/random/mersenne_twister.hpp>
#include <boost/assign/ptr_map_inserter.hpp>
#include <boost/random/variate_generator.hpp>

#include <map>
#include <vector>
#include <iomanip>
#include <sstream>
#include <cstring>
#include <iomanip>
#include <signal.h>
#include <algorithm>

#ifndef ALPS_NG_DEPRECATED_SIMULATION_HPP
#define ALPS_NG_DEPRECATED_SIMULATION_HPP

namespace alps {
    namespace ng {

        class deprecated_simulation : public singe_simulation {
            public:
                deprecated_simulation(parameters_type const & p, std::size_t seed_offset = 0)
                    : singe_simulation(p, seed_offset)
                    , parms(make_alps_parameters(p))
                    , measurements(results)
                    , random_01(random)
                {}

                double fraction_completed() const { return work_done(); }

                virtual double work_done() const = 0;

                virtual void dostep() = 0;

                double random_real(double a = 0., double b = 1.) { return a + b * random(); }

                virtual void do_update() {
                    dostep();
                }

                virtual void do_measurements() {}

            protected:
                Parameters parms;
                ObservableSet & measurements;
                boost::variate_generator<boost::mt19937, boost::uniform_real<> > & random_01;

            private:
                static Parameters make_alps_parameters(parameters_type const & s) {
                    Parameters p;
                    for (parameters_type::const_iterator it = s.begin(); it != s.end(); ++it)
// TODO: why does static_cast<std::string>(it->second) not work?
                        p.push_back(it->first, it->second.operator std::string());
                    return p;
                }
        };

    }
}

#endif
