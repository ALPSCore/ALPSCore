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

#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>

#include <numpy/arrayobject.h>

// !! This is very ugly - any ideas how to improve?
#include <alps/hdf5/hdf5.cpp>

namespace alps { 
    namespace python {
        namespace hdf5 {
          
            void import_numpy() {
                static bool inited = false;
                if (!inited) {
                    import_array();  
                    boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
                    inited = true;
                }
            }

            template<typename T> boost::python::str filename(T & self) {
                return boost::python::str(self.filename());
            }

            template<typename T> boost::python::list extent(T & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::size_t> children = self.extent(path);
                for (std::vector<std::size_t>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(*it);
                return result;
            }

            template<typename T> boost::python::list list_children(T & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::string> children = self.list_children(path);
                for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(boost::python::str(*it));
                return result;
            }

            template<typename T> boost::python::list list_attributes(T & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::string> children = self.list_attributes(path);
                for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(boost::python::str(*it));
                return result;
            }

            void write(alps::hdf5::oarchive & self, std::string const & path, boost::python::object const & data) {
                using alps::make_pvp;
                import_numpy();
                if (false);
                #define PYHDF5_CHECK_SCALAR(T, F)                                                                                                          \
                    else if ( F (data.ptr()))                                                                                                              \
                        self << make_pvp(path, boost::python::extract< T >(data)());
                PYHDF5_CHECK_SCALAR(int, PyInt_CheckExact)
                PYHDF5_CHECK_SCALAR(long, PyLong_CheckExact)
                PYHDF5_CHECK_SCALAR(double, PyFloat_CheckExact)
                PYHDF5_CHECK_SCALAR(std::complex<double>, PyComplex_CheckExact)
                PYHDF5_CHECK_SCALAR(std::string, PyString_CheckExact)
                #undef PYHDF5_CHECK_SCALAR
                else {
                    import_numpy();
                    if (PyArray_Check(data.ptr())) {
                        if (!PyArray_ISCONTIGUOUS(data.ptr()))
                            throw std::runtime_error("numpy array is not continous");
                        else if (!PyArray_ISNOTSWAPPED(data.ptr()))
                            throw std::runtime_error("numpy array is not native");
                        #define PYHDF5_CHECK_NUMPY(T, N)                                                                                                   \
                            else if (PyArray_TYPE(data.ptr()) == N)                                                                                        \
                                self << make_pvp(                                                                                                          \
                                      path                                                                                                                 \
                                    , static_cast< T const *>(PyArray_DATA(data.ptr()))                                                                    \
                                    , std::vector<std::size_t>(PyArray_DIMS(data.ptr()), PyArray_DIMS(data.ptr()) + PyArray_NDIM(data.ptr()))              \
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
                    } else
                        throw std::runtime_error("unsupported type");
                }
            }

            template<typename T> boost::python::object read_scalar(alps::hdf5::iarchive & self, std::string const & path) {
                T data;
                self >> make_pvp(path, data);
                return boost::python::object(data);
            }

            template<> boost::python::object read_scalar<std::string>(alps::hdf5::iarchive & self, std::string const & path) {
                std::string data;
                self >> make_pvp(path, data);
                return boost::python::str(data);
            }

            template<typename T> boost::python::object read_numpy(alps::hdf5::iarchive & self, std::string const & path, int type) {
                import_numpy();
                std::pair<T *, std::vector<std::size_t> > data(NULL, self.extent(path));
                std::vector<npy_intp> npextent(data.second.begin(), data.second.end());
                std::size_t len = std::accumulate(data.second.begin(), data.second.end(), std::size_t(1), std::multiplies<std::size_t>());
                boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(npextent.size(), &npextent.front(), type)));
                if (len) {
                    data.first = new T[len];
                    try {
                        self >> make_pvp(path, data);
                        memcpy(PyArray_DATA(obj.ptr()), data.first, PyArray_ITEMSIZE(obj.ptr()) * PyArray_SIZE(obj.ptr()));
                        delete[] data.first;
                    } catch (...) {
                        delete[] data.first;
                        throw;
                    }
                }
                return obj;
            }

            boost::python::object dispatch_read(alps::hdf5::iarchive & self, std::string const & path) {
                if (self.is_scalar(path)) {
                    if (self.is_string(path))
                        return read_scalar<std::string>(self, path);
                    else if (self.is_int(path))
                        return read_scalar<int>(self, path);
                    else if (self.is_uint(path))
                        return read_scalar<unsigned int>(self, path);
                    else if (self.is_long(path))
                        return read_scalar<long>(self, path);
                    else if (self.is_ulong(path))
                        return read_scalar<unsigned long>(self, path);
                    else if (self.is_longlong(path))
                        return read_scalar<long long>(self, path);
                    else if (self.is_ulonglong(path))
                        return read_scalar<unsigned long long>(self, path);
                    else if (self.is_float(path) || self.is_double(path))
                        return read_scalar<double>(self, path);
                    else if (self.is_complex(path))
                        return read_scalar<std::complex<double> >(self, path);
                    else
                        std::runtime_error("Unsupported type.");
                } else if (self.is_string(path)) {
                    if (self.dimensions(path) != 1)
                        std::runtime_error("More than 1 Dimension is not supported.");
                    boost::python::list result;
                    std::vector<std::string> data;
                    self >> make_pvp(path, data);
                    for (std::vector<std::string>::const_iterator it = data.begin(); it != data.end(); ++it)
                         result.append(boost::python::str(*it));
                    return result;
                } else if (self.is_int(path))
                    return read_numpy<int>(self, path, PyArray_INT);
                else if (self.is_uint(path))
                    return read_numpy<unsigned int>(self, path, PyArray_INT);
                else if (self.is_long(path))
                    return read_numpy<long>(self, path, PyArray_LONG);
                else if (self.is_ulong(path))
                    return read_numpy<unsigned long>(self, path, PyArray_ULONG);
                else if (self.is_longlong(path))
                    return read_numpy<long long>(self, path, PyArray_LONGLONG);
                else if (self.is_ulonglong(path))
                    return read_numpy<unsigned long long>(self, path, PyArray_ULONGLONG);
                else if (self.is_float(path) || self.is_double(path))
                    return read_numpy<double>(self, path, PyArray_DOUBLE);
                else if (self.is_complex(path))
                    return read_numpy<std::complex<double> >(self, path, PyArray_CDOUBLE);
                else
                    std::runtime_error("Unsupported type.");
            }

        }
    }
}

const char archive_constructor_docstring[] =
"the constructor takes a file path as argument";

const char filename_docstring[] = 
"the (read-only) file name of the archive";

const char is_group_docstring[] = 
"returns True if the given path is a group in the HDF5 file";

const char is_data_docstring[] = 
"returns True if the given path points to data in the HDF5 file";

const char is_attribute_docstring[] = 
"returns True if the given path points to an attribute in the HDF5 file";

const char is_scalar_docstring[] = 
"returns True if the given path points to scalar data in the HDF5 file";

const char is_null_docstring[] = 
"returns True if the given path points to an empty (null) node in the HDF5 file";

const char extent_docstring[] = 
"returns a list of the extents along any of the dimensions of an array "
"at the specified path in the HDF5 file";

const char dimensions_docstring[] = 
"returns the number of dimensions of an array at the specified path in the HDF5 file";

const char list_children_docstring[] =
"returns a list with the relative paths of all child nodes "
"relative to a given path in the HDF5 file";

const char list_attr_docstring[] =
"returns a list with the names of all the attributes "
"of a given path in the HDF5 file";

const char write_docstring[] =
"writes an object into the specified path in the HDF5 file.\n"
"Currently supported types are scalar types and numpy arrays of them.";

const char read_docstring[] =
"reads an object from the specified path in the HDF5 file.\n"
"Currently supported types are scalar types and numpy arrays of them.";

BOOST_PYTHON_MODULE(pyhdf5_c) {
    using namespace boost::python;
    docstring_options doc_options(true);
    doc_options.disable_cpp_signatures();
  
    class_<alps::hdf5::oarchive>(
          "oArchive", 
          "an archive class to write HDF5 files", 
          boost::python::init<std::string>(archive_constructor_docstring)
        )
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::oarchive>)
        .add_property("filename", &alps::python::hdf5::filename<alps::hdf5::oarchive>,filename_docstring)
        .def("is_group", &alps::hdf5::oarchive::is_group,is_group_docstring)
        .def("is_data", &alps::hdf5::oarchive::is_data,is_data_docstring)
        .def("is_attribute", &alps::hdf5::oarchive::is_attribute,is_attribute_docstring)
        .def("extent", &alps::python::hdf5::extent<alps::hdf5::oarchive>,extent_docstring)
        .def("dimensions", &alps::hdf5::oarchive::dimensions,dimensions_docstring)
        .def("is_scalar", &alps::hdf5::oarchive::is_scalar,is_scalar_docstring)
        .def("is_null", &alps::hdf5::oarchive::is_null,is_null_docstring)
        .def("list_children", &alps::python::hdf5::list_children<alps::hdf5::oarchive>,list_children_docstring)
        .def("list_attributes", &alps::python::hdf5::list_attributes<alps::hdf5::oarchive>,list_attr_docstring)
        .def("write", &alps::python::hdf5::write, write_docstring)
    ;

    class_<alps::hdf5::iarchive>(
          "iArchive", 
          "an archive class to read HDF5 files", 
           boost::python::init<std::string>(archive_constructor_docstring)
        )
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::iarchive>)
        .add_property("filename", &alps::python::hdf5::filename<alps::hdf5::iarchive>,filename_docstring)
        .def("is_group", &alps::hdf5::iarchive::is_group,is_group_docstring)
        .def("is_data", &alps::hdf5::iarchive::is_data,is_data_docstring)
        .def("is_attribute", &alps::hdf5::iarchive::is_attribute,is_attribute_docstring)
        .def("extent", &alps::python::hdf5::extent<alps::hdf5::iarchive>,extent_docstring)
        .def("dimensions", &alps::hdf5::iarchive::dimensions,dimensions_docstring)
        .def("is_scalar", &alps::hdf5::iarchive::is_scalar,is_scalar_docstring)
        .def("is_null", &alps::hdf5::iarchive::is_null,is_null_docstring)
        .def("list_children", &alps::python::hdf5::list_children<alps::hdf5::iarchive>,list_children_docstring)
        .def("list_attributes", &alps::python::hdf5::list_attributes<alps::hdf5::iarchive>,list_attr_docstring)
        .def("read", &alps::python::hdf5::dispatch_read,read_docstring)
    ;

}
