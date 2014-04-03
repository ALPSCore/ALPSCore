/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_DETAIL_EXTRACT_FROM_PYOBJECT_HPP
#define ALPS_NGS_DETAIL_EXTRACT_FROM_PYOBJECT_HPP

#if defined(ALPS_HAVE_PYTHON)

    #include <alps/ngs/cast.hpp>
    #include <alps/ngs/config.hpp>

    #include <alps/ngs/boost_python.hpp>
    #include <alps/ngs/detail/type_wrapper.hpp>
    #include <alps/ngs/detail/numpy_import.hpp>
    #include <alps/ngs/detail/get_numpy_type.hpp>

    #include <boost/python/list.hpp>
    #include <boost/python/dict.hpp>

    namespace alps {
        namespace detail {

            // TODO: move to file and use it in pyngshdf5
            template<typename T> void extract_from_pyobject(T & visitor, boost::python::object const & data) {
                import_numpy();
                std::string dtype = data.ptr()->ob_type->tp_name;
                if (dtype == "bool") visitor(boost::python::extract<bool>(data)());
                else if (dtype == "int") visitor(boost::python::extract<int>(data)());
                else if (dtype == "long") visitor(boost::python::extract<long>(data)());
                else if (dtype == "float") visitor(boost::python::extract<double>(data)());
                else if (dtype == "complex") visitor(boost::python::extract<std::complex<double> >(data)());
                else if (dtype == "str") visitor(boost::python::extract<std::string>(data)());
                else if (dtype == "list") visitor(boost::python::list(data));
                else if (dtype == "dict") visitor(boost::python::dict(data));
                else if (dtype == "numpy.str") visitor(boost::python::call_method<std::string>(data.ptr(), "__str__"));
                else if (dtype == "numpy.bool") visitor(boost::python::call_method<bool>(data.ptr(), "__bool__"));
                else if (dtype == "numpy.int8") visitor(static_cast<boost::int8_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.int16") visitor(static_cast<boost::int16_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.int32") visitor(static_cast<boost::int32_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.int64") visitor(static_cast<boost::int64_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.uint8") visitor(static_cast<boost::uint8_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.uint16") visitor(static_cast<boost::uint16_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.uint32") visitor(static_cast<boost::uint32_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.uint64") visitor(static_cast<boost::uint64_t>(boost::python::call_method<long>(data.ptr(), "__long__")));
                else if (dtype == "numpy.float32") visitor(static_cast<float>(boost::python::call_method<double>(data.ptr(), "__float__")));
                else if (dtype == "numpy.float64") visitor(static_cast<double>(boost::python::call_method<double>(data.ptr(), "__float__")));
                else if (dtype == "numpy.complex64") 
                    visitor(std::complex<float>(
                          boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("real").ptr()), "__float__")
                        , boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("imag").ptr()), "__float__")
                    ));
                else if (dtype == "numpy.complex128")
                    visitor(std::complex<double>(
                          boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("real").ptr()), "__float__")
                        , boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("imag").ptr()), "__float__")
                    ));
                else if (dtype == "numpy.ndarray") {
                    PyArrayObject * ptr = (PyArrayObject *)data.ptr();
                    if (!PyArray_Check(ptr))
                        throw std::runtime_error("invalid numpy data" + ALPS_STACKTRACE);
                    else if (!PyArray_ISNOTSWAPPED(ptr))
                        throw std::runtime_error("numpy array is not native" + ALPS_STACKTRACE);
                    else if (!(ptr = PyArray_GETCONTIGUOUS(ptr)))
                        throw std::runtime_error("numpy array cannot be converted to continous array" + ALPS_STACKTRACE);
                    #define ALPS_NGS_EXTRACT_FROM_PYOBJECT_CHECK_NUMPY(T)                                                                               \
                        else if (PyArray_DESCR(ptr)->type_num == detail::get_numpy_type(type_wrapper< T >::type()))                                     \
                            visitor(                                                                                                                    \
                                  static_cast< T const *>(PyArray_DATA(ptr))                                                                            \
                                , std::vector<std::size_t>(PyArray_DIMS(ptr), PyArray_DIMS(ptr) + PyArray_NDIM(ptr))                                    \
                            );
                    ALPS_NGS_FOREACH_NATIVE_NUMPY_TYPE(ALPS_NGS_EXTRACT_FROM_PYOBJECT_CHECK_NUMPY)
                    #undef ALPS_NGS_EXTRACT_FROM_PYOBJECT_CHECK_NUMPY
                    else
                        throw std::runtime_error("Unknown numpy element type: " + cast<std::string>(PyArray_DESCR(ptr)->type_num) + ALPS_STACKTRACE);
                    Py_DECREF((PyObject *)ptr);
                } else
                    throw std::runtime_error("Unsupported type: " + dtype + ALPS_STACKTRACE);
            }
        }
    }

#endif

#endif
