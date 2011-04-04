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

#define PY_ARRAY_UNIQUE_SYMBOL pyhdf5_PyArrayHandle

#include <alps/ngs/mchdf5.hpp>
#include <alps/ngs/mchdf5/pair.hpp>
#include <alps/ngs/mchdf5/vector.hpp>
#include <alps/ngs/mchdf5/complex.hpp>

#include <alps/python/make_copy.hpp>

#include <boost/python.hpp>
#include <boost/scoped_ptr.hpp>

#include <numpy/arrayobject.h>

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

            boost::python::str filename(alps::hdf5::archive const & self) {
                return boost::python::str(self.get_filename());
            }

            boost::python::list extent(alps::hdf5::archive & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::size_t> children = self.extent(path);
                for (std::vector<std::size_t>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(*it);
                return result;
            }

            boost::python::list list_children(alps::hdf5::archive & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::string> children = self.list_children(path);
                for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(boost::python::str(*it));
                return result;
            }

            boost::python::list list_attributes(alps::hdf5::archive & self, std::string const & path) {
                boost::python::list result;
                std::vector<std::string> children = self.list_attributes(path);
                for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it)
                    result.append(boost::python::str(*it));
                return result;
            }

            void write(alps::hdf5::archive & self, std::string const & path, boost::python::object const & data) {
                using alps::make_pvp;
                import_numpy();
                if (false);
                #define PYHDF5_CHECK_SCALAR(T, F)                                                                                                          \
                    else if ( F (data.ptr()))                                                                                                              \
							\
							\
							{\
							std::cerr << (alps::hdf5::has_complex_elements<T>::value ? "true" : "false") << " " << (boost::is_same<T, std::complex<double> >::value ? "true" : "false") << std::endl;\
							\
							\
							\
                        self << make_pvp(path, boost::python::extract< T >(data)());			\
						\
												\
												}\
												\
						
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
							\
							\
							{\
							std::cerr << (alps::hdf5::has_complex_elements<T>::value ? "true" : "false") << std::endl;\
							\
							\
							\
                                self << make_pvp(                                                                                                          \
                                      path                                                                                                                 \
                                    , std::make_pair(                                                                                                      \
                                          static_cast< T const *>(PyArray_DATA(data.ptr()))                                                                \
                                        , std::vector<std::size_t>(PyArray_DIMS(data.ptr()), PyArray_DIMS(data.ptr()) + PyArray_NDIM(data.ptr()))          \
                                    )                                                                                                                      \
                                );				\
												\
												}\
												\
												
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
                        PYHDF5_CHECK_NUMPY(long long, PyArray_LONGLONG)
                        PYHDF5_CHECK_NUMPY(unsigned long long, PyArray_ULONGLONG)
                        PYHDF5_CHECK_NUMPY(float, PyArray_FLOAT)
                        PYHDF5_CHECK_NUMPY(double, PyArray_DOUBLE)
                        PYHDF5_CHECK_NUMPY(long double, PyArray_LONGDOUBLE)
                        PYHDF5_CHECK_NUMPY(std::complex<float>, PyArray_CFLOAT)
                        PYHDF5_CHECK_NUMPY(std::complex<double>, PyArray_CDOUBLE)
                        PYHDF5_CHECK_NUMPY(std::complex<long double>, PyArray_CLONGDOUBLE)
                        #undef PYHDF5_CHECK_NUMPY
                        else
                            throw std::runtime_error("unsupported numpy array type");
                    } else
                        throw std::runtime_error("unsupported type");
                }
            }

            template<typename T> boost::python::object read_scalar(alps::hdf5::archive & self, std::string const & path) {
                T data;
                self >> make_pvp(path, data);
                return boost::python::object(data);
            }

            template<> boost::python::object read_scalar<std::string>(alps::hdf5::archive & self, std::string const & path) {
                std::string data;
                self >> make_pvp(path, data);
                return boost::python::str(data);
            }

            template<typename T> boost::python::object read_numpy(alps::hdf5::archive & self, std::string const & path, int type) {
                import_numpy();
                std::vector<std::size_t> extent(self.extent(path));
                std::vector<npy_intp> npextent(extent.begin(), extent.end());
                std::size_t len = std::accumulate(extent.begin(), extent.end(), std::size_t(1), std::multiplies<std::size_t>());
                boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(npextent.size(), &npextent.front(), type)));
                if (len) {
                    boost::scoped_ptr<T> raw(new T[len]);
                    std::pair<T *, std::vector<std::size_t> > data(raw.get(), extent);
                    self >> make_pvp(path, data);
                    memcpy(PyArray_DATA(obj.ptr()), raw.get(), PyArray_ITEMSIZE(obj.ptr()) * PyArray_SIZE(obj.ptr()));
                }
                return obj;
            }

            boost::python::object dispatch_read(alps::hdf5::archive & self, std::string const & path) {
                if (self.is_scalar(path)) {
                    if (self.is_datatype<std::string>(path))
                        return read_scalar<std::string>(self, path);
                    else if (self.is_datatype<int>(path))
                        return read_scalar<int>(self, path);
                    else if (self.is_datatype<unsigned int>(path))
                        return read_scalar<unsigned int>(self, path);
                    else if (self.is_datatype<long>(path))
                        return read_scalar<long>(self, path);
                    else if (self.is_datatype<unsigned long>(path))
                        return read_scalar<unsigned long>(self, path);
                    else if (self.is_datatype<long long>(path))
                        return read_scalar<long long>(self, path);
                    else if (self.is_datatype<unsigned long long>(path))
                        return read_scalar<unsigned long long>(self, path);
                    else if (self.is_datatype<float>(path))
                        return read_scalar<float>(self, path);
                    else if (self.is_datatype<double>(path))
                        return read_scalar<double>(self, path);
                    else if (self.is_complex(path))
                        return read_scalar<std::complex<double> >(self, path);
                    else
                        std::runtime_error("Unsupported type.");
                } else if (self.is_datatype<std::string>(path)) {
                    if (self.dimensions(path) != 1)
                        std::runtime_error("More than 1 Dimension is not supported.");
                    boost::python::list result;
                    std::vector<std::string> data;
                    self >> make_pvp(path, data);
                    for (std::vector<std::string>::const_iterator it = data.begin(); it != data.end(); ++it)
                         result.append(boost::python::str(*it));
                    return result;
                } else if (self.is_datatype<int>(path))
                    return read_numpy<int>(self, path, PyArray_INT);
                else if (self.is_datatype<unsigned int>(path))
                    return read_numpy<unsigned int>(self, path, PyArray_UINT);
                else if (self.is_datatype<long>(path))
                    return read_numpy<long>(self, path, PyArray_LONG);
                else if (self.is_datatype<unsigned long>(path))
                    return read_numpy<unsigned long>(self, path, PyArray_ULONG);
                else if (self.is_datatype<long long>(path))
                    return read_numpy<long long>(self, path, PyArray_LONGLONG);
                else if (self.is_datatype<unsigned long long>(path))
                    return read_numpy<unsigned long long>(self, path, PyArray_ULONGLONG);
                else if (self.is_datatype<float>(path) || self.is_datatype<double>(path))
                    return read_numpy<double>(self, path, PyArray_DOUBLE);
                else if (self.is_complex(path))
                    return read_numpy<std::complex<double> >(self, path, PyArray_CDOUBLE);
                else {
                    std::runtime_error("Unsupported type.");
                    return boost::python::object();
                }
            }

        }
    }
}

const char archive_constructor_docstring[] =
"the constructor takes a file path the opening mode (read: 0, write: 1, write compressed: 3) as argument";

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

const char is_complex_docstring[] = 
"returns True if the given path points to complex data in the HDF5 file";

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

const char create_group_docstring[] =
"create a group at the specified path in the HDF5 file.";

const char delete_data_docstring[] =
"delete a dataset at the specified path in the HDF5 file.\n";

const char delete_group_docstring[] =
"delete a group at the specified path in the HDF5 file.\n";

const char delete_attribute_docstring[] =
"delete an attribute at the specified path in the HDF5 file.\n";

BOOST_PYTHON_MODULE(pyhdf5_c) {
    using namespace boost::python;
    docstring_options doc_options(true);
    doc_options.disable_cpp_signatures();

    class_<alps::hdf5::archive>(
          "archive", 
          "an archive class to read and write HDF5 files", 
          boost::python::init<std::string, std::size_t>(archive_constructor_docstring)
    )
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::archive>)
        .add_property("filename", &alps::python::hdf5::filename, filename_docstring)
        .def("is_group", &alps::hdf5::archive::is_group, is_group_docstring)
        .def("is_data", &alps::hdf5::archive::is_data, is_data_docstring)
        .def("is_attribute", &alps::hdf5::archive::is_attribute, is_attribute_docstring)
        .def("extent", &alps::python::hdf5::extent, extent_docstring)
        .def("dimensions", &alps::hdf5::archive::dimensions, dimensions_docstring)
        .def("is_scalar", &alps::hdf5::archive::is_scalar, is_scalar_docstring)
        .def("is_complex", &alps::hdf5::archive::is_complex, is_complex_docstring)
        .def("is_null", &alps::hdf5::archive::is_null, is_null_docstring)
        .def("list_children", &alps::python::hdf5::list_children, list_children_docstring)
        .def("list_attributes", &alps::python::hdf5::list_attributes, list_attr_docstring)
        .def("read", &alps::python::hdf5::dispatch_read, read_docstring)
        .def("write", &alps::python::hdf5::write, write_docstring)
        .def("create_group", &alps::hdf5::archive::create_group, create_group_docstring)
        .def("delete_data", &alps::hdf5::archive::delete_data, delete_data_docstring)
        .def("delete_group", &alps::hdf5::archive::delete_group, delete_group_docstring)
        .def("delete_attribute", &alps::hdf5::archive::delete_attribute, delete_attribute_docstring)
    ;

}
