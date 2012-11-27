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
#include <alps/ngs/cast.hpp>
#include <alps/ngs/mcresult.hpp>

#include <alps/ngs/boost_python.hpp>
#include <alps/ngs/detail/numpy_import.ipp>

#include <alps/python/make_copy.hpp>


namespace alps {
    namespace detail {

        template <typename T> std::string short_print_python(T const & value) {
            return cast<std::string>(value);
        }

        template <typename T> std::string short_print_python(std::vector<T> const & value) {
            switch (value.size()) {
                case 0: 
                    return "[]";
                case 1: 
                    return "[" + short_print_python(value.front()) + "]";
                case 2: 
                    return "[" + short_print_python(value.front()) + "," + short_print_python(value.back()) + "]";
                default: 
                    return "[" + short_print_python(value.front()) + ",.." + short_print_python(value.size()) + "..," + short_print_python(value.back()) + "]";
            }
        }

        boost::python::str mcresult_print(alps::mcresult const & self) {
            if (self.count() == 0)
                return boost::python::str("No Measurements");
            else if (self.is_type<double>())
                return boost::python::str(
                      short_print_python(self.mean<double>()) + "(" + short_print_python(self.count()) + ") "
                    + "+/-" + short_print_python(self.error<double>()) + " "
                    + short_print_python(self.bins<double>()) + "#" + short_print_python(self.bin_size())
                );
            else if (self.is_type<std::vector<double> >())
                return boost::python::str(
                      short_print_python(self.mean<std::vector<double> >()) + "(" + short_print_python(self.count()) + ") "
                    + "+/-" + short_print_python(self.error<std::vector<double> >()) + " "
                    + short_print_python(self.bins<std::vector<double> >()) + "#" + short_print_python(self.bin_size())
                );
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
        }

        boost::python::object mcresult_vector2np(std::vector<double> const & data) {
            import_numpy();
            npy_intp size = data.size();
            boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &size, PyArray_DOUBLE)));
            if (size)
                memcpy(PyArray_DATA(obj.ptr()), &data.front(), PyArray_ITEMSIZE(obj.ptr()) * PyArray_SIZE(obj.ptr()));
            return obj;
        }

        boost::python::object mcresult_mean(alps::mcresult const & self) {
            if (self.is_type<double>())
                return boost::python::object(self.mean<double>());
            else if (self.is_type<std::vector<double> >())
                return mcresult_vector2np(self.mean<std::vector<double> >());
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
            return boost::python::object();
        }

        boost::python::object mcresult_error(alps::mcresult const & self) {
            if (self.is_type<double>())
                return boost::python::object(self.error<double>());
            else if (self.is_type<std::vector<double> >())
                return mcresult_vector2np(self.error<std::vector<double> >());
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
            return boost::python::object();
        }

        boost::python::object mcresult_tau(alps::mcresult const & self) {
            if (self.is_type<double>())
                return boost::python::object(self.tau<double>());
            else if (self.is_type<std::vector<double> >())
                return mcresult_vector2np(self.tau<std::vector<double> >());
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
            return boost::python::object();
        }

        boost::python::object mcresult_variance(alps::mcresult const & self) {
            if (self.is_type<double>())
                return boost::python::object(self.variance<double>());
            else if (self.is_type<std::vector<double> >())
                return mcresult_vector2np(self.variance<std::vector<double> >());
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
            return boost::python::object();
        }

        boost::python::object mcresult_bins(alps::mcresult const & self) {
            if (self.is_type<double>())
                return mcresult_vector2np(self.bins<double>());
//          else if (self.is_type<std::vector<double> >())
//              return mcresult_vector2np(self.bins<std::vector<double> >());
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
            return boost::python::object();
        }
        
        alps::mcresult observable2result_export(alps::mcobservable const & obs) {
            return alps::mcresult(obs);
        }

    }
}

BOOST_PYTHON_MODULE(pyngsresult_c) {
    using boost::python::self;
    using namespace alps;

    boost::python::def("observable2result", &alps::detail::observable2result_export);

    boost::python::class_<alps::mcresult>(
        "result",
        boost::python::init<boost::python::optional<alps::mcresult> >()
    )
        .def("__repr__", &alps::detail::mcresult_print)
        .def("__deepcopy__", &alps::python::make_copy<alps::mcresult >)
        .def("__abs__", static_cast<alps::mcresult(*)(alps::mcresult)>(&abs))
        .def("__pow__", static_cast<alps::mcresult(*)(alps::mcresult, double)>(&pow))

        .add_property("mean", &alps::detail::mcresult_mean)
        .add_property("error", &alps::detail::mcresult_error)
        .add_property("tau", &alps::detail::mcresult_tau)
        .add_property("variance", &alps::detail::mcresult_variance)
        .add_property("bins", &alps::detail::mcresult_bins)
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
        .def(self + alps::mcresult())
        .def(alps::mcresult() + self)
        .def(self + double())
        .def(double() + self)
        .def(self - alps::mcresult())
        .def(alps::mcresult() - self)
        .def(self - double())
        .def(double() - self)
        .def(self * alps::mcresult())
        .def(alps::mcresult() * self)
        .def(self * double())
        .def(double() * self)
        .def(self / alps::mcresult())
        .def(alps::mcresult() / self)
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
// asinh, aconsh and atanh are not part of C++03 standard
//        .def("asinh", static_cast<alps::mcresult(*)(alps::mcresult)>(&asinh))
//        .def("acosh", static_cast<alps::mcresult(*)(alps::mcresult)>(&acosh))
//        .def("atanh", static_cast<alps::mcresult(*)(alps::mcresult)>(&atanh))

        .def("save", &alps::mcresult::save)
        .def("load", &alps::mcresult::load)
    ;

}
