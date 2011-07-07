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

#define PY_ARRAY_UNIQUE_SYMBOL pyngsresult_PyArrayHandle

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/mcresult.hpp>

#include <alps/python/make_copy.hpp>

#include <alps/ngs/boost_python.hpp>

namespace alps {
    namespace detail {

        boost::python::str mcresult_print(alps::mcresult const & self) {
            boost::python::str str;
/*
            return boost::python::str(boost::python::str(self.mean()) + " +/- " + boost::python::str(self.error()));
        }

        template <typename T> boost::python::str mcresult_print(alps::alea::mcdata<std::vector<T> > const & self) {
*/
/*
            for (typename alps::alea::mcdata<std::vector<T> >::const_iterator it = self.begin(); it != self.end(); ++it)
                str += print_mcdata(*it) + (it + 1 != self.end() ? "\n" : "");
*/
            return str;
        }

    }
}

BOOST_PYTHON_MODULE(pyngsresult_c) {
    using boost::python::self;
    using namespace alps;

    boost::python::class_<alps::mcresult>(
        "result",
        boost::python::no_init
    )
        .def("__repr__", &alps::detail::mcresult_print)
        .def("__deepcopy__", &alps::python::make_copy<alps::mcresult >)
        .def("__abs__", static_cast<alps::mcresult(*)(alps::mcresult)>(&abs))
        .def("__pow__", static_cast<alps::mcresult(*)(alps::mcresult, double)>(&pow))

//        .add_property("mean", static_cast<double(*)(alps::mcresult const &)>(&alps::python::wrap_mean),mean_docstring)
//        .add_property("error", static_cast<double(*)(alps::mcresult const &)>(&alps::python::wrap_error),error_docstring)
//        .add_property("tau", static_cast<double(*)(alps::mcresult const &)>(&alps::python::wrap_tau),tau_docstring)
//        .add_property("variance", static_cast<double(*)(alps::mcresult const &)>(&alps::python::wrap_variance),variance_docstring)
//        .add_property("bins", static_cast<numeric::array(*)(alps::mcresult const &)>(&alps::python::wrap_bins),bins_docstring)
        .add_property("count", &alps::mcresult::count)

        .def(+self)
        .def(-self)
        .def(self += alps::mcresult())
        .def(self += double())
        .def(self -= alps::mcresult())
        .def(self -= double())
        .def(self *= alps::mcresult())
        .def(self *= double())
        .def(self /= alps::mcresult())
        .def(self /= double())
//        .def(self + alps::mcresult())
//        .def(alps::mcresult() + self)
        .def(self + double())
        .def(double() + self)
//        .def(self - alps::mcresult())
//        .def(alps::mcresult() - self)
        .def(self - double())
        .def(double() - self)
//        .def(self * alps::mcresult())
//        .def(alps::mcresult() * self)
        .def(self * double())
        .def(double() * self)
//        .def(self / alps::mcresult())
//        .def(alps::mcresult() / self)
        .def(self / double())
        .def(double() / self)

        .def("sq", static_cast<alps::mcresult(*)(alps::mcresult)>(&sq))
        .def("cb", static_cast<alps::mcresult(*)(alps::mcresult)>(&cb))
        .def("sqrt", static_cast<alps::mcresult(*)(alps::mcresult)>(&sqrt))
        .def("cbrt", static_cast<alps::mcresult(*)(alps::mcresult)>(&cbrt))
        .def("exp", static_cast<alps::mcresult(*)(alps::mcresult)>(&exp))
        .def("log", static_cast<alps::mcresult(*)(alps::mcresult)>(&log))
        .def("sin", static_cast<alps::mcresult(*)(alps::mcresult)>(&sin))
        .def("cos", static_cast<alps::mcresult(*)(alps::mcresult)>(&cos))
        .def("tan", static_cast<alps::mcresult(*)(alps::mcresult)>(&tan))
        .def("asin", static_cast<alps::mcresult(*)(alps::mcresult)>(&asin))
        .def("acos", static_cast<alps::mcresult(*)(alps::mcresult)>(&acos))
        .def("atan", static_cast<alps::mcresult(*)(alps::mcresult)>(&atan))
        .def("sinh", static_cast<alps::mcresult(*)(alps::mcresult)>(&sinh))
        .def("cosh", static_cast<alps::mcresult(*)(alps::mcresult)>(&cosh))
        .def("tanh", static_cast<alps::mcresult(*)(alps::mcresult)>(&tanh))
        .def("asinh", static_cast<alps::mcresult(*)(alps::mcresult)>(&asinh))
        .def("acosh", static_cast<alps::mcresult(*)(alps::mcresult)>(&acosh))
        .def("atanh", static_cast<alps::mcresult(*)(alps::mcresult)>(&atanh))
        .def("save", &alps::mcresult::save)
        .def("load", &alps::mcresult::load)
    ;

}
