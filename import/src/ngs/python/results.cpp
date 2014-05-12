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
