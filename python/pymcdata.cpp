/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Lukas Gamper <gamperl@gmail.com>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
*
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

/* $Id: pyalea.cpp 3520 2010-04-09 16:49:53Z tamama $ */

#define PY_ARRAY_UNIQUE_SYMBOL pyalea_PyArrayHandle

#include <alps/alea/mcdata.hpp>
#include <alps/python/make_copy.hpp>
#include <alps/python/numpy_array.hpp>

#include <boost/python.hpp>

namespace alps { 
    namespace alea {

        template <typename T>
        mcdata<T>::mcdata(boost::python::object const & mean)
                            : count_(1)
                            , binsize_(0)
                            , max_bin_number_(0)
                            , data_is_analyzed_(true)
                            , jacknife_bins_valid_(true)
                            , cannot_rebin_(false)
                        {
                            alps::python::numpy::convert(mean, mean_);
                        }

        template <typename T>
        mcdata<T>::mcdata(boost::python::object const & mean, boost::python::object const & error)
                            : count_(1)
                            , binsize_(0)
                            , max_bin_number_(0)
                            , data_is_analyzed_(true)
                            , jacknife_bins_valid_(true)
                            , cannot_rebin_(false)
                        {
                            alps::python::numpy::convert(mean, mean_);
                            alps::python::numpy::convert(error, error_);
                        }
    }

    namespace python {

        template<typename T> std::size_t size(alps::alea::mcdata<T> & data) {
            return data.mean().size();
        }

        template<typename T> boost::python::object get_item(boost::python::back_reference<alps::alea::mcdata<T> &> data, PyObject* i) {
            if (PySlice_Check(i)) {
                PySliceObject * slice = static_cast<PySliceObject *>(static_cast<void *>(i));
                if (Py_None != slice->step) {
                    PyErr_SetString(PyExc_IndexError, "slice step size not supported.");
                    boost::python::throw_error_already_set();
                }
                long from = (Py_None == slice->start ? 0 : boost::python::extract<long>(slice->start)());
                if (from < 0)
                    from += size(data.get());
                from = std::max<long>(std::min<long>(from, size(data.get())), 0);
                long to = (Py_None == slice->stop ? 0 : boost::python::extract<long>(slice->stop)());
                if (to < 0)
                    to += size(data.get());
                to = std::max<long>(std::min<long>(to, size(data.get())), 0);
                if (from > to)
                    return boost::python::object(alps::alea::mcdata<T>());
                else
                    return boost::python::object(alps::alea::mcdata<T>(data.get(), from, to));
            } else {
                long index = 0;
                if (boost::python::extract<long>(i).check()) {
                    index = boost::python::extract<long>(i)();
                    if (index < 0)
                        index += size(data.get());
                    if (index >= (long)size(data.get()) || index < 0) {
                        PyErr_SetString(PyExc_IndexError, "Index out of range");
                        boost::python::throw_error_already_set();
                    }
                } else {
                    PyErr_SetString(PyExc_TypeError, "Invalid index type");
                    boost::python::throw_error_already_set();
                }
                return boost::python::object(alps::alea::mcdata<typename T::value_type>(data.get(), index));
            }
        }

        template<typename T> bool contains(alps::alea::mcdata<T> & data, PyObject* key) {
            boost::python::extract<alps::alea::mcdata<typename T::value_type> const &> x(key);
            if (x.check())
                return std::find(data.begin(), data.end(), x()) != data.end();
            else {
                boost::python::extract<alps::alea::mcdata<typename T::value_type> > x(key);
                if (x.check())
                    return std::find(data.begin(), data.end(), x()) != data.end();
                else
                    return false;
            }
        }

        #define ALPS_PY_MCDATA_WRAPPER(member_name)                                                                                                               \
            template <class T> typename alps::alea::mcdata<T>::result_type wrap_ ## member_name(alps::alea::mcdata<T> const & value) {                            \
                return value. member_name ();                                                                                                                     \
            }                                                                                                                                                     \
            template <class T> boost::python::numeric::array wrap_ ## member_name(alps::alea::mcdata<T> const & value) {                                          \
                return alps::python::numpy::convert(value. member_name ());                                                                                       \
            }

        ALPS_PY_MCDATA_WRAPPER(mean)
        ALPS_PY_MCDATA_WRAPPER(error)
        ALPS_PY_MCDATA_WRAPPER(tau)
        ALPS_PY_MCDATA_WRAPPER(variance)
        ALPS_PY_MCDATA_WRAPPER(bins)
        #undef ALPS_PY_MCDATA_WRAPPER

        template <typename T> boost::python::str print_mcdata(alps::alea::mcdata<T> const & self) {
            return boost::python::str(boost::python::str(self.mean()) + " +/- " + boost::python::str(self.error()));
        }

        template <typename T> boost::python::str print_mcdata(alps::alea::mcdata<std::vector<T> > const & self) {
            boost::python::str str;
            for (typename alps::alea::mcdata<std::vector<T> >::const_iterator it = self.begin(); it != self.end(); ++it)
                str += print_mcdata(*it) + (it + 1 != self.end() ? "\n" : "");
            return str;
        }

    }
}

using namespace alps::alea;
using namespace boost::python;

const char mcdata_docstring[] =
"This class is used to evaluate Monte Carlo data and functions on them. "
"Besides the documented functions and properties, the class supports the "
"arithmetic operations +, -, *, /, +=, -=, *=, /=, and the functions "
"abs, acos, acosh, asin, asinh, atan, atanh, cb, cbrt, cos, cosh, exp, log, "
"pow, sin, sinh, sq, sqrt, tan and tanh.";


const char init_docstring[] =
"Optionally the constructor takes one argument for the mean values, "
"and a second optional argumnt for the error.";

const char save_docstring[] =
"Saves the object into the HDF5 file specified as the first string argument, "
"using as observable name the second string argument.";

const char load_docstring[] =
"Loads the object into from HDF5 file specified as the first string argument, "
"using as observable name the second string argument.";

const char mean_docstring[] =
"the mean value of all measurements recorded.";

const char error_docstring[] =
"the error of all measurements recorded.";

const char tau_docstring[] =
"the autocorrelation time estimate of the recorded measurements.";

const char variance_docstring[] =
"the variance of all measurements recorded.";

const char count_docstring[] =
"the number of measurements recorded.";

const char bins_docstring[] =
"the number of bins recorded for a jackknife analysis.";


BOOST_PYTHON_MODULE(pymcdata_c) {

    class_<mcdata<double> >("MCScalarData", mcdata_docstring,init<optional<double, double> >(init_docstring))
        .add_property("mean", static_cast<double(*)(mcdata<double> const &)>(&alps::python::wrap_mean),mean_docstring)
        .add_property("error", static_cast<double(*)(mcdata<double> const &)>(&alps::python::wrap_error),error_docstring)
        .add_property("tau", static_cast<double(*)(mcdata<double> const &)>(&alps::python::wrap_tau),tau_docstring)
        .add_property("variance", static_cast<double(*)(mcdata<double> const &)>(&alps::python::wrap_variance),variance_docstring)
        .add_property("bins", static_cast<numeric::array(*)(mcdata<double> const &)>(&alps::python::wrap_bins),bins_docstring)
        .add_property("count", &mcdata<double>::count,count_docstring)
        .def("__repr__", static_cast<str(*)(mcdata<double> const &)>(&alps::python::print_mcdata))
        .def("__deepcopy__", &alps::python::make_copy<mcdata<double> >)
        .def("__abs__", static_cast<mcdata<double>(*)(mcdata<double>)>(&abs))
        .def("__pow__", static_cast<mcdata<double>(*)(mcdata<double>, mcdata<double>::element_type)>(&pow))
        .def(+self)
        .def(-self)
        .def(self += mcdata<double>())
        .def(self += double())
        .def(self -= mcdata<double>())
        .def(self -= double())
        .def(self *= mcdata<double>())
        .def(self *= double())
        .def(self /= mcdata<double>())
        .def(self /= double())
        .def(self + mcdata<double>())
        .def(mcdata<double>() + self)
        .def(self + double())
        .def(double() + self)
        .def(self - mcdata<double>())
        .def(mcdata<double>() - self)
        .def(self - double())
        .def(double() - self)
        .def(self * mcdata<double>())
        .def(mcdata<double>() * self)
        .def(self * mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() * self)
        .def(self * double())
        .def(double() * self)
        .def(self / mcdata<double>())
        .def(mcdata<double>() / self)
        .def(self / double())
        .def(double() / self)
        .def("sq", static_cast<mcdata<double>(*)(mcdata<double>)>(&sq))
        .def("cb", static_cast<mcdata<double>(*)(mcdata<double>)>(&cb))
        .def("sqrt", static_cast<mcdata<double>(*)(mcdata<double>)>(&sqrt))
        .def("cbrt", static_cast<mcdata<double>(*)(mcdata<double>)>(&cbrt))
        .def("exp", static_cast<mcdata<double>(*)(mcdata<double>)>(&exp))
        .def("log", static_cast<mcdata<double>(*)(mcdata<double>)>(&log))
        .def("sin", static_cast<mcdata<double>(*)(mcdata<double>)>(&sin))
        .def("cos", static_cast<mcdata<double>(*)(mcdata<double>)>(&cos))
        .def("tan", static_cast<mcdata<double>(*)(mcdata<double>)>(&tan))
        .def("asin", static_cast<mcdata<double>(*)(mcdata<double>)>(&asin))
        .def("acos", static_cast<mcdata<double>(*)(mcdata<double>)>(&acos))
        .def("atan", static_cast<mcdata<double>(*)(mcdata<double>)>(&atan))
        .def("sinh", static_cast<mcdata<double>(*)(mcdata<double>)>(&sinh))
        .def("cosh", static_cast<mcdata<double>(*)(mcdata<double>)>(&cosh))
        .def("tanh", static_cast<mcdata<double>(*)(mcdata<double>)>(&tanh))
        .def("asinh", static_cast<mcdata<double>(*)(mcdata<double>)>(&asinh))
        .def("acosh", static_cast<mcdata<double>(*)(mcdata<double>)>(&acosh))
        .def("atanh", static_cast<mcdata<double>(*)(mcdata<double>)>(&atanh))
        .def("save", static_cast<void(mcdata<double>::*)(std::string const &, std::string const &) const>(&mcdata<double>::save),save_docstring)
        .def("load", static_cast<void(mcdata<double>::*)(std::string const &, std::string const &)>(&mcdata<double>::load),load_docstring)
    ;

    class_<mcdata<std::vector<double> > >("MCVectorData", mcdata_docstring, init<optional<object, object> >(init_docstring))
        .def("__len__", static_cast<std::size_t(*)(alps::alea::mcdata<std::vector<double> > &)>(&alps::python::size))
        .def("__getitem__", static_cast<object(*)(back_reference<alps::alea::mcdata<std::vector<double> > & >, PyObject *)>(&alps::python::get_item))
        .def("__contains__", static_cast<bool(*)(alps::alea::mcdata<std::vector<double> > &, PyObject *)>(&alps::python::contains))
        .add_property("mean", static_cast<numeric::array(*)(mcdata<std::vector<double> > const &)>(&alps::python::wrap_mean),mean_docstring)
        .add_property("error", static_cast<numeric::array(*)(mcdata<std::vector<double> > const &)>(&alps::python::wrap_error),error_docstring)
        .add_property("tau", static_cast<numeric::array(*)(mcdata<std::vector<double> > const &)>(&alps::python::wrap_tau),tau_docstring)
        .add_property("variance", static_cast<numeric::array(*)(mcdata<std::vector<double> > const &)>(&alps::python::wrap_variance),variance_docstring)
        .add_property("bins", static_cast<numeric::array(*)(mcdata<std::vector<double> > const &)>(&alps::python::wrap_bins),bins_docstring)
        .add_property("count", &mcdata<std::vector<double> >::count,count_docstring)
        .def("__repr__", static_cast<str(*)(mcdata<std::vector<double> > const &)>(&alps::python::print_mcdata))
        .def("__deepcopy__", &alps::python::make_copy<mcdata<std::vector<double> > >)
        .def("__abs__", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&abs))
        .def("__pow__", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >, mcdata<double>::element_type)>(&pow))
        .def(+self)
        .def(-self)
        .def(self == mcdata<std::vector<double> >())
        .def(self += mcdata<std::vector<double> >())
        .def(self += std::vector<double>())
        .def(self -= mcdata<std::vector<double> >())
        .def(self -= std::vector<double>())
        .def(self *= mcdata<std::vector<double> >())
        .def(self *= std::vector<double>())
        .def(self /= mcdata<std::vector<double> >())
        .def(self /= std::vector<double>())
        .def(self + mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() + self)
        .def(self + std::vector<double>())
        .def(std::vector<double>() + self)
        .def(self - mcdata<std::vector<double> >())
        .def(mcdata<std::vector<double> >() - self)
        .def(self - std::vector<double>())
        .def(std::vector<double>() - self)
        .def(self * mcdata<std::vector<double> >())
        .def(self * mcdata<double>())
        .def(mcdata<double>() * self)
        .def(mcdata<std::vector<double> >() * self)
        .def(self * std::vector<double>())
        .def(std::vector<double>() * self)
        .def(self / mcdata<std::vector<double> >())
        .def(self / mcdata<double>())
        .def(mcdata<std::vector<double> >() / self)
        .def(self / std::vector<double>())
        .def(std::vector<double>() / self)
        .def(self + double())
        .def(double() + self)
        .def(self - double())
        .def(double() - self)
        .def(self * double())
        .def(double() * self)
        .def(self / double())
        .def(double() / self)
        .def("sq", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sq))
        .def("cb", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cb))
        .def("sqrt", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sqrt))
        .def("cbrt", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cbrt))
        .def("exp", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&exp))
        .def("log", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&log))
        .def("sin", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sin))
        .def("cos", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cos))
        .def("tan", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&tan))
        .def("asin", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&asin))
        .def("acos", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&acos))
        .def("atan", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&atan))
        .def("sinh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&sinh))
        .def("cosh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&cosh))
        .def("tanh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&tanh))
        .def("asinh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&asinh))
        .def("acosh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&acosh))
        .def("atanh", static_cast<mcdata<std::vector<double> >(*)(mcdata<std::vector<double> >)>(&atanh))
        .def("save", static_cast<void(mcdata<std::vector<double> >::*)(std::string const &, std::string const &) const>(&mcdata<std::vector<double> >::save),save_docstring)
        .def("load", static_cast<void(mcdata<std::vector<double> >::*)(std::string const &, std::string const &)>(&mcdata<std::vector<double> >::load),load_docstring)
    ;
}
