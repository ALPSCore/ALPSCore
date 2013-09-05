/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
 *                              Matthias Troyer <troyer@comp-phys.org>             *
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

#define PY_ARRAY_UNIQUE_SYMBOL pyngsobservables_PyArrayHandle

#include <alps/hdf5/archive.hpp>
#include <alps/ngs/mcobservable.hpp>
#include <alps/ngs/mcobservables.hpp>
#include <alps/ngs/observablewrappers.hpp>

#include <alps/ngs/boost_python.hpp>
#include <boost/python/suite/indexing/map_indexing_suite.hpp>

void mcobservables_load(alps::mcobservables & self, alps::hdf5::archive & ar, std::string const & path) {
    std::string current = ar.get_context();
    ar.set_context(path);
    self.load(ar);
    ar.set_context(current);
}

void createRealObservable(alps::mcobservables & self, std::string const & name, boost::uint32_t binnum = 0) {
    self << alps::ngs::RealObservable(name, binnum);
}
BOOST_PYTHON_FUNCTION_OVERLOADS(createRealObservable_overloads, createRealObservable, 2, 3)

void createRealVectorObservable(alps::mcobservables & self, std::string const & name, boost::uint32_t binnum = 0) {
    self << alps::ngs::RealVectorObservable(name, binnum);
}
BOOST_PYTHON_FUNCTION_OVERLOADS(createRealVectorObservable_overloads, createRealVectorObservable, 2, 3)

void addObservable(alps::mcobservables & self, boost::python::object obj) {
    boost::python::call_method<void>(obj.ptr(), "addToObservables", boost::ref(self));
}

BOOST_PYTHON_MODULE(pyngsobservables_c) {

    boost::python::class_<alps::mcobservables>(
        "observables",
//        boost::python::no_init      // Tamama removes this line: Reason: this adds an __init__ method which always raises a Python Runtime exception. 
        boost::python::init<>()     // Tamama add this line.
    )
        .def(boost::python::map_indexing_suite<alps::mcobservables>())
        .def("reset", &alps::mcobservables::reset)
        .def("save", &alps::mcobservables::save)
        .def("load", &mcobservables_load)
        .def("__lshift__", &addObservable)
        .def("createRealObservable", &createRealObservable, createRealObservable_overloads())
        .def("createRealVectorObservable", &createRealVectorObservable, createRealVectorObservable_overloads())
        // TODO: implement!
/*
        .def("createRealVectorObservable", &alps::mcobservables::create_RealVectorObservable)
        .def("createSimpleRealObservable", &alps::mcobservables::create_SimpleRealObservable)
        .def("createSimpleRealVectorObservable", &alps::mcobservables::create_SimpleRealVectorObservable)
        .def("createSignedRealObservable", &alps::mcobservables::create_SignedRealObservable)
        .def("createSignedRealVectorObservable", &alps::mcobservables::create_SignedRealVectorObservable)
        .def("createSignedSimpleRealObservable", &alps::mcobservables::create_SignedSimpleRealObservable)
        .def("createSignedSimpleRealVectorObservable", &alps::mcobservables::create_SignedSimpleRealVectorObservable)
*/
    ;

}
