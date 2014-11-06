/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define PY_ARRAY_UNIQUE_SYMBOL pyngsrandom_PyArrayHandle

#include <alps/utilities/boost_python.hpp>

#include <alps/mc/random01.hpp>

// #include <alps/python/make_copy.hpp>

BOOST_PYTHON_MODULE(pyrandom01_c) {

    boost::python::class_<alps::random01>(
        "random01",
        boost::python::init<boost::python::optional<int> >()
    )
    	// TODO: implement ...
        // .def("__deepcopy__",  &alps::python::make_copy<alps::random01>)
        .def("__call__", static_cast<alps::random01::result_type(alps::random01::*)()>(&alps::random01::operator()))
        .def("save", &alps::random01::save)
        .def("load", &alps::random01::load)
    ;
}
