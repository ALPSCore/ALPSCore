/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_API_HPP
#define ALPS_API_HPP

#include <alps/config.h>
#include <alps/params.hpp>
#include <alps/accumulator/accumulator.hpp>

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

    #ifdef ALPS_USE_NEW_ALEA
        ALPS_DECL void save_results(alps::accumulator::accumulator_set const & observables, params const & params, boost::filesystem::path const & filename, std::string const & path);
        ALPS_DECL void save_results(alps::accumulator::result_set const & results, params const & params, boost::filesystem::path const & filename, std::string const & path);
    #endif

    //ALPS_DECL void save_results(mcresults const & results, params const & params, boost::filesystem::path const & filename, std::string const & path);

    //ALPS_DECL void save_results(mcobservables const & observables, params const & params, boost::filesystem::path const & filename, std::string const & path);

    template<typename C, typename P> void broadcast(C const & c, P & p, int r = 0) {
        p.broadcast(c, r);
    }

}

#endif
