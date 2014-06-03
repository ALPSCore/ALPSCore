/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define PY_ARRAY_UNIQUE_SYMBOL pyngsresults_PyArrayHandle

#include <alps/hdf5.hpp>
#include <alps/ngs/mcresults.hpp>

#include <alps/ngs/boost_python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

namespace alps {
    namespace detail {

        std::string mcresults_print(alps::mcresults & self) {
            std::stringstream sstr;
            sstr << self;
            return sstr.str();
        }

        void mcresults_load(alps::mcresults & self, alps::hdf5::archive & ar, std::string const & path) {
            std::string current = ar.get_context();
            ar.set_context(path);
            self.load(ar);
            ar.set_context(current);
        }
    }
}

BOOST_PYTHON_MODULE(pyngsresults_c) {
    boost::python::class_<alps::mcresults>(
        "results",
        boost::python::no_init
    )
        .def(boost::python::map_indexing_suite<alps::mcresults>())
        .def("__str__", &alps::detail::mcresults_print)
        .def("save", &alps::mcresults::save)
        .def("load", &alps::detail::mcresults_load)
    ;

}
