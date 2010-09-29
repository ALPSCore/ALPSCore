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
#include <alps/ng/alea.hpp>
#include <alps/parameter.h>
#include <alps/ng/boost.hpp>
#include <alps/ng/signal.hpp>
#include <alps/ng/options.hpp>
#include <alps/ng/parameters.hpp>

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

#ifndef ALPS_NG_API_HPP
#define ALPS_NG_API_HPP

namespace alps {
    namespace ng {

        template<typename S> struct result_names_type {
            typedef typename S::result_names_type type;
        };

        template<typename S> struct results_type {
            typedef typename S::results_type type;
        };

        template<typename S> struct parameters_type {
            typedef typename S::parameters_type type;
        };

        template<typename S> typename result_names_type<S>::type result_names(S const & s) {
            return s.result_names();
        }

        template<typename S> typename result_names_type<S>::type unsaved_result_names(S const & s) {
            return s.unsaved_result_names();
        }

        template<typename S> typename results_type<S>::type collect_results(S const & s) {
            return s.collect_results();
        }

        template<typename S> typename results_type<S>::type collect_results(S const & s, typename result_names_type<S>::type const & names) {
            return s.collect_results(names);
        }

        template<typename S> typename results_type<S>::type collect_results(S const & s, std::string const & name) {
            return collect_results(s, typename result_names_type<S>::type(1, name));
        }

        template<typename S> double fraction_completed(S const & s) {
            return s.fraction_completed();
        }

    }
}

#endif
