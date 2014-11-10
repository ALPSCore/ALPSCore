/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define PY_ARRAY_UNIQUE_SYMBOL pyapi_PyArrayHandle

// this must be first
#include <alps/utilities/boost_python.hpp>

#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>

namespace alps {
    namespace detail {

        void save_results_export(
        	  alps::results_type<alps::mcbase>::type const & results
        	, alps::parameters_type<alps::mcbase>::type const & par, alps::hdf5::archive & ar
        	, std::string const & path
        ) {
            ar["/parameters"] << par;
            if (results.size())
                ar[path] << results;
        }
    }
}

BOOST_PYTHON_MODULE(pyapi_c) {

    boost::python::def("collectResults", static_cast<alps::results_type<alps::mcbase>::type (*)(alps::mcbase const &)>(&alps::collect_results<alps::mcbase>));

    // boost::python::def("saveResults", &alps::detail::save_results_export);

}
