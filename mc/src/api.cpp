/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/api.hpp>
#include <alps/hdf5/archive.hpp>

#include <boost/filesystem.hpp>

namespace alps {

    namespace detail {
        template<typename R, typename P> void save_results_impl(R const & results, P const & params, boost::filesystem::path const & filename, std::string const & path) {
            if (results.size()) {
                hdf5::archive ar(filename.string(), "w");
                ar["/parameters"] << params;
                ar[path] << results;
            }
        }
    }

    void save_results(alps::accumulators::result_set const & results, params const & params, boost::filesystem::path const & filename, std::string const & path) {
        detail::save_results_impl(results, params, filename, path);
    }

    void save_results(alps::accumulators::accumulator_set const & observables, params const & params, boost::filesystem::path const & filename, std::string const & path) {
        detail::save_results_impl(observables, params, filename, path);
    }

}
