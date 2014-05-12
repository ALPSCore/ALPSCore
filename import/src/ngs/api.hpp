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

#ifndef ALPS_NGS_API_HPP
#define ALPS_NGS_API_HPP

#include <alps/ngs/config.hpp>
#include <alps/ngs/params.hpp>
#include <alps/ngs/mcresults.hpp>
#include <alps/ngs/mcobservables.hpp>

#ifdef ALPS_NGS_USE_NEW_ALEA
    #include <alps/ngs/accumulator/accumulator.hpp>
#endif

#include <boost/filesystem/path.hpp>

#include <string>

namespace alps {

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

    #ifdef ALPS_NGS_USE_NEW_ALEA
        ALPS_DECL void save_results(alps::accumulator::accumulator_set const & observables, params const & params, boost::filesystem::path const & filename, std::string const & path);
        ALPS_DECL void save_results(alps::accumulator::result_set const & results, params const & params, boost::filesystem::path const & filename, std::string const & path);
    #endif

    ALPS_DECL void save_results(mcresults const & results, params const & params, boost::filesystem::path const & filename, std::string const & path);

    ALPS_DECL void save_results(mcobservables const & observables, params const & params, boost::filesystem::path const & filename, std::string const & path);

    template<typename C, typename P> void broadcast(C const & c, P & p, int r = 0) {
        p.broadcast(c, r);
    }

}

#endif
