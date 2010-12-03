/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Lukas Gamper <gamperl@gmail.com>,
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

/* $Id: pyalea.cpp 3520 2010-04-09 16:49:53Z gamperl $ */

#define PY_ARRAY_UNIQUE_SYMBOL pyhdf5_PyArrayHandle

#define ALPS_HDF5_NO_LEXICAL_CAST

#include <alps/hdf5.hpp>
#include <alps/python/make_copy.hpp>
#include <alps/python/numpy_array.hpp>

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include <numpy/arrayobject.h>

namespace alps { 
    namespace python {
        namespace hdf5 {

            template<typename T> boost::python::str filename(T & self, std::string const & path) {
                return boost::python::str(self.filename());
            }


            template<typename T> boost::python::list list_children(T & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::string> children = self.list_children(path);
                for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(boost::python::str(*it));
                return result;
            }

            template<typename T> boost::python::list list_attr(T & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::string> children = self.list_attr(path);
                for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(boost::python::str(*it));
                return result;
            }

            template<typename T> void write_scalar(alps::hdf5::oarchive & self, std::string const & path, T data) {
                using alps::make_pvp;
                self << make_pvp(path, data);
            }

            template<typename T> void write_numpy(alps::hdf5::oarchive & self, std::string const & path, T const * data, std::vector<std::size_t> const & v) {
                using alps::make_pvp;
                alps::python::numpy::import();
                self << make_pvp(path, data, v);
            }

            void dispatch_write(alps::hdf5::oarchive & self, std::string const & path, boost::python::object const & data) {
                if (false);
                #define PYHDF5_CHECK_SCALAR(T)                                                                                                             \
                    else if (boost::python::extract< T >(data).check())                                                                                    \
                        write_scalar(self, path, boost::python::extract< T >(data)());
                PYHDF5_CHECK_SCALAR(bool)
                PYHDF5_CHECK_SCALAR(char)
                PYHDF5_CHECK_SCALAR(signed char)
                PYHDF5_CHECK_SCALAR(unsigned char)
                PYHDF5_CHECK_SCALAR(short)
                PYHDF5_CHECK_SCALAR(unsigned short)
                PYHDF5_CHECK_SCALAR(int)
                PYHDF5_CHECK_SCALAR(unsigned int)
                PYHDF5_CHECK_SCALAR(long)
                PYHDF5_CHECK_SCALAR(unsigned long)
                #ifndef BOOST_NO_LONG_LONG
                    PYHDF5_CHECK_SCALAR(long long)
                    PYHDF5_CHECK_SCALAR(unsigned long long)
                #endif
                PYHDF5_CHECK_SCALAR(float)
                PYHDF5_CHECK_SCALAR(double)
                PYHDF5_CHECK_SCALAR(long double)
                PYHDF5_CHECK_SCALAR(std::complex<float>)
                PYHDF5_CHECK_SCALAR(std::complex<double>)
                PYHDF5_CHECK_SCALAR(std::complex<long double>)
                #undef PYHDF5_CHECK_SCALAR
                if (PyArray_Check(data.ptr())) {
                    if (!PyArray_ISCONTIGUOUS(data.ptr()))
                        throw std::runtime_error("numpy array is not continous");
                    else if (!PyArray_ISNOTSWAPPED(data.ptr()))
                        throw std::runtime_error("numpy array is not native");
                    #define PYHDF5_CHECK_NUMPY(T, N)                                                                                                       \
                        else if (PyArray_TYPE(data.ptr()) == N)                                                                                            \
                            write_numpy(                                                                                                                   \
                                  self                                                                                                                     \
                                , path                                                                                                                     \
                                , static_cast< T const *>(PyArray_DATA(data.ptr()))                                                                        \
                                , std::vector<std::size_t>(PyArray_DIMS(data.ptr()), PyArray_DIMS(data.ptr()) + PyArray_NDIM(data.ptr()))                  \
                            );
                    PYHDF5_CHECK_NUMPY(bool, PyArray_BOOL)
                    PYHDF5_CHECK_NUMPY(char, PyArray_CHAR)
                    PYHDF5_CHECK_NUMPY(signed char, PyArray_BYTE)
                    PYHDF5_CHECK_NUMPY(unsigned char, PyArray_UBYTE)
                    PYHDF5_CHECK_NUMPY(short, PyArray_SHORT)
                    PYHDF5_CHECK_NUMPY(unsigned short, PyArray_USHORT)
                    PYHDF5_CHECK_NUMPY(int, PyArray_INT)
                    PYHDF5_CHECK_NUMPY(unsigned int, PyArray_UINT)
                    PYHDF5_CHECK_NUMPY(long, PyArray_LONG)
                    PYHDF5_CHECK_NUMPY(unsigned long, PyArray_ULONG)
                    #ifndef BOOST_NO_LONG_LONG
                        PYHDF5_CHECK_NUMPY(long long, PyArray_LONGLONG)
                        PYHDF5_CHECK_NUMPY(unsigned long long, PyArray_ULONGLONG)
                    #endif
                    PYHDF5_CHECK_NUMPY(float, PyArray_FLOAT)
                    PYHDF5_CHECK_NUMPY(double, PyArray_DOUBLE)
                    PYHDF5_CHECK_NUMPY(long double, PyArray_LONGDOUBLE)
                    PYHDF5_CHECK_NUMPY(std::complex<float>, PyArray_CFLOAT)
                    PYHDF5_CHECK_NUMPY(std::complex<double>, PyArray_CDOUBLE)
                    PYHDF5_CHECK_NUMPY(std::complex<long double>, PyArray_CLONGDOUBLE)
                    #undef PYHDF5_CHECK_SCALAR
                    else
                        throw std::runtime_error("unsupported numpy array type");
                }
                throw std::runtime_error("unsupported type");
            }

            // TODO: allow more types
            boost::python::object dispatch_read(alps::hdf5::iarchive & self, std::string const & path) {
                if (self.is_scalar(path)) {
                    if (self.is_string(path)) {
                        std::string data;
                        self >> make_pvp(path, data);
                        return boost::python::str(data);
                    } else {
                        double data;
                        self >> make_pvp(path, data);
                        return boost::python::object(data);
                    }
                } else if (self.is_string(path)) {
                    if (self.dimensions(path) != 1)
                        std::runtime_error("More than 1 Dimension is not supported.");
                    boost::python::list result;
                    std::vector<std::string> data;
                    self >> make_pvp(path, data);
                    for (std::vector<std::string>::const_iterator it = data.begin(); it != data.end(); ++it)
                         result.append(boost::python::str(*it));
                    return result;
                } else {
                    alps::python::numpy::import();
                    std::pair<double *, std::vector<std::size_t> > data(NULL, self.extent(path));
                    std::vector<npy_intp> npextent(data.second.begin(), data.second.end());
                    boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(npextent.size(), &npextent.front(), PyArray_DOUBLE)));
                    data.first = new double[std::accumulate(data.second.begin(), data.second.end(), std::size_t(1), std::multiplies<std::size_t>())];
                    try {
                        self >> make_pvp(path, data);
                        memcpy(PyArray_DATA(obj.ptr()), data.first, PyArray_ITEMSIZE(obj.ptr()) * PyArray_SIZE(obj.ptr()));
                        delete[] data.first;
                    } catch (...) {
                        delete[] data.first;
                        throw;
                    }
                    return obj;
                }
            }


        }
    }
}

BOOST_PYTHON_MODULE(pyhdf5_c) {

    boost::python::class_<alps::hdf5::oarchive>("oArchive", boost::python::init<std::string>())
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::oarchive>)
        .def("filename", &alps::python::hdf5::filename<alps::hdf5::oarchive>)
        .def("is_group", &alps::hdf5::oarchive::is_group)
        .def("is_data", &alps::hdf5::oarchive::is_data)
        .def("is_attribute", &alps::hdf5::oarchive::extent)
        .def("extent", &alps::hdf5::oarchive::dimensions)
        .def("dimensions", &alps::hdf5::oarchive::dimensions)
        .def("is_scalar", &alps::hdf5::oarchive::is_scalar)
        .def("is_null", &alps::hdf5::oarchive::is_null)
        .def("list_children", &alps::python::hdf5::list_children<alps::hdf5::oarchive>)
        .def("list_attr", &alps::python::hdf5::list_attr<alps::hdf5::oarchive>)
        .def("write", &alps::python::hdf5::dispatch_write)
    ;

    boost::python::class_<alps::hdf5::iarchive>("iArchive", boost::python::init<std::string>())
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::iarchive>)
        .def("filename", &alps::python::hdf5::filename<alps::hdf5::iarchive>)
        .def("is_group", &alps::hdf5::iarchive::is_group)
        .def("is_data", &alps::hdf5::iarchive::is_data)
        .def("is_attribute", &alps::hdf5::iarchive::extent)
        .def("extent", &alps::hdf5::iarchive::dimensions)
        .def("dimensions", &alps::hdf5::iarchive::dimensions)
        .def("is_scalar", &alps::hdf5::iarchive::is_scalar)
        .def("is_null", &alps::hdf5::iarchive::is_null)
        .def("list_children", &alps::python::hdf5::list_children<alps::hdf5::iarchive>)
        .def("list_attr", &alps::python::hdf5::list_attr<alps::hdf5::iarchive>)
        .def("read", &alps::python::hdf5::dispatch_read)
    ;

}
