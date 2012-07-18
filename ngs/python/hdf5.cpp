/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Lukas Gamper <gamperl@gmail.com>                   *
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

#define PY_ARRAY_UNIQUE_SYMBOL pyngshdf5_PyArrayHandle

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/hdf5/pair.hpp>
#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/hdf5/vector.hpp>
#include <alps/ngs/hdf5/complex.hpp>

#include <alps/ngs/boost_python.hpp>
#include <alps/ngs/lib/numpy_import.ipp>

#include <alps/python/make_copy.hpp>

#include <boost/scoped_ptr.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/to_python_converter.hpp>

#include <string>
#include <iterator>
#include <stdexcept>

namespace alps {
    namespace detail {

        struct std_string_to_python {
            static PyObject* convert(std::string const & value) {
                return boost::python::incref(boost::python::str(value).ptr());
            }
        };

        struct std_vector_string_to_python {
            static PyObject* convert(std::vector<std::string> const & value) {
                boost::python::list result;
                for (std::vector<std::string>::const_iterator it = value.begin(); it != value.end(); ++it)
                    result.append(boost::python::str(*it));
                return boost::python::incref(result.ptr());
            }
        };

        // TODO: move to hdf5/<name>.hpp
        // make python objects serializable
        class hdf5_archive_export : public alps::hdf5::archive {
            using alps::hdf5::archive::extent;
            using alps::hdf5::archive::is_datatype;

            public:
            
                hdf5_archive_export(std::string const & filename, std::string mode)
                    : alps::hdf5::archive(filename, mode)
                {}

                boost::python::str py_filename() {
                    return boost::python::str(get_filename());
                }

                boost::python::str py_context() {
                    return boost::python::str(get_context());
                }

                boost::python::list py_extent(std::string const & path) {
                    boost::python::list result;
                    std::vector<std::size_t> ext = extent(path);
                    if (is_complex(path)) {
                        if (ext.size() > 1)
                            ext.pop_back();
                        else
                            ext.back() = 1;
                    }
                    for (std::vector<std::size_t>::const_iterator it = ext.begin(); it != ext.end(); ++it)
                        result.append(*it);
                    return result;
                }

                void py_save(std::string const & path, boost::python::object const & data) {
                    import_numpy();
                    std::string dtype = data.ptr()->ob_type->tp_name;
                    if (dtype == "bool") py_save_scalar(path, boost::python::extract<bool>(data)());
                    else if (dtype == "int") py_save_scalar(path, boost::python::extract<int>(data)());
                    else if (dtype == "long") py_save_scalar(path, boost::python::extract<long>(data)());
                    else if (dtype == "float") py_save_scalar(path, boost::python::extract<double>(data)());
                    else if (dtype == "complex") py_save_scalar(path, boost::python::extract<std::complex<double> >(data)());
                    else if (dtype == "str") py_save_scalar(path, boost::python::extract<std::string>(data)());
                    else if (dtype == "list") py_save_list(path, boost::python::extract<boost::python::list>(data)());
                    else if (dtype == "dict") py_save_dict(path, boost::python::extract<boost::python::dict>(data)());
                    else if (dtype == "numpy.ndarray") py_save_numpy(path, boost::python::extract<boost::python::numeric::array>(data)());
                    else if (dtype == "numpy.str") py_save_scalar(path, boost::python::call_method<std::string>(data.ptr(), "__str__"));
                    else if (dtype == "numpy.bool") py_save_scalar(path, boost::python::call_method<bool>(data.ptr(), "__bool__"));
                    else if (dtype == "numpy.int8") py_save_scalar<boost::int8_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.int16") py_save_scalar<boost::int16_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.int32") py_save_scalar<boost::int32_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.int64") py_save_scalar<boost::int64_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.uint8") py_save_scalar<boost::uint8_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.uint16") py_save_scalar<boost::uint16_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.uint32") py_save_scalar<boost::uint32_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.uint64") py_save_scalar<boost::uint64_t>(path, boost::python::call_method<long>(data.ptr(), "__long__"));
                    else if (dtype == "numpy.float32") py_save_scalar<float>(path, boost::python::call_method<double>(data.ptr(), "__float__"));
                    else if (dtype == "numpy.float64") py_save_scalar<double>(path, boost::python::call_method<double>(data.ptr(), "__float__"));
                     else if (dtype == "numpy.complex64") py_save_scalar(path, std::complex<float>(
                          boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("real").ptr()), "__float__")
                        , boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("imag").ptr()), "__float__")
                    ));
                     else if (dtype == "numpy.complex128") py_save_scalar(path, std::complex<double>(
                          boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("real").ptr()), "__float__")
                        , boost::python::call_method<double>(PyObject_GetAttr(data.ptr(), boost::python::str("imag").ptr()), "__float__")
                    ));
                    else
                        throw std::runtime_error("Unsupported type: " + dtype + ALPS_STACKTRACE);
                }

                boost::python::object py_load(std::string const & path) {
                    import_numpy();
                    if (is_group(path)) {
                        std::vector<std::string> list = list_children(path);
                        bool is_list = true;
                        for (std::vector<std::string>::const_iterator it = list.begin(); is_list && it != list.end(); ++it)
                            for (std::string::const_iterator jt = it->begin(); is_list && jt != it->end(); ++jt)
                                if (std::string("1234567890").find_first_of(*jt) == std::string::npos || alps::cast<unsigned>(*jt) > list.size() - 1)
                                    is_list = false;
                        if (is_list && list.size()) {
                            std::vector<boost::python::object> result;
                            for (std::size_t i = 0; i < list.size(); ++i)
                                result.push_back(py_load(path + "/" + alps::cast<std::string>(i)));
                            return boost::python::list(result);
                        } else {
                            boost::python::dict result;
                            for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it)
                                boost::python::call_method<void>(result.ptr(), "__setitem__", *it, py_load(path + "/" + encode_segment(*it)));
                            return result;
                        }
                    } else if (is_scalar(path) || (is_datatype<double>(path) && is_complex(path) && extent(path).size() == 1 && extent(path)[0] == 2)) {
                        if (is_datatype<std::string>(path))
                            return load_scalar<std::string>(path);
                        else if (is_datatype<bool>(path))
                            return load_scalar<bool>(path);
                        else if (is_datatype<int>(path))
                            return load_scalar<int>(path);
                        else if (is_datatype<unsigned int>(path))
                            return load_scalar<unsigned int>(path);
                        else if (is_datatype<long>(path))
                            return load_scalar<long>(path);
                        else if (is_datatype<unsigned long>(path))
                            return load_scalar<unsigned long>(path);
                        else if (is_datatype<long long>(path))
                            return load_scalar<long long>(path);
                        else if (is_datatype<unsigned long long>(path))
                            return load_scalar<unsigned long long>(path);
                        else if (is_datatype<float>(path))
                            return load_scalar<float>(path);
                        else if (is_datatype<double>(path) && is_complex(path))
                            return load_scalar<std::complex<double> >(path);
                        else if (is_datatype<double>(path))
                            return load_scalar<double>(path);
                        else
                            throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
                    } else if (is_datatype<std::string>(path)) {
                        if (dimensions(path) != 1)
                            throw std::runtime_error("More than 1 Dimension is not supported." + ALPS_STACKTRACE);
                        boost::python::list result;
                        std::vector<std::string> data;
                        static_cast<alps::hdf5::archive &>(*this) >> make_pvp(path, data);
                        for (std::vector<std::string>::const_iterator it = data.begin(); it != data.end(); ++it)
                             result.append(boost::python::str(*it));
                        return result;
                    } else if (is_datatype<int>(path))
                        return load_numpy<int>(path, PyArray_INT);
                    else if (is_datatype<unsigned int>(path))
                        return load_numpy<unsigned int>(path, PyArray_UINT);
                    else if (is_datatype<long>(path))
                        return load_numpy<long>(path, PyArray_LONG);
                    else if (is_datatype<unsigned long>(path))
                        return load_numpy<unsigned long>(path, PyArray_ULONG);
                    else if (is_datatype<long long>(path))
                        return load_numpy<long long>(path, PyArray_LONGLONG);
                    else if (is_datatype<unsigned long long>(path))
                        return load_numpy<unsigned long long>(path, PyArray_ULONGLONG);
                    else if (is_datatype<float>(path) && is_complex(path))
                        return load_numpy<std::complex<float> >(path, PyArray_CFLOAT);
                    else if (is_datatype<double>(path) && is_complex(path))
                        return load_numpy<std::complex<double> >(path, PyArray_CDOUBLE);
                    else if (is_datatype<long double>(path) && is_complex(path))
                        return load_numpy<std::complex<long double> >(path, PyArray_CLONGDOUBLE);
                    else if (is_datatype<float>(path))
                        return load_numpy<std::complex<float> >(path, PyArray_FLOAT);
                    else if (is_datatype<double>(path))
                        return load_numpy<std::complex<double> >(path, PyArray_DOUBLE);
                    else if (is_datatype<long double>(path))
                        return load_numpy<std::complex<long double> >(path, PyArray_LONGDOUBLE);
                    else
                        throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
                    return boost::python::object();
                }

            private:

                template<typename T> boost::python::object load_scalar(std::string const & path) {
                    T data;
                    static_cast<alps::hdf5::archive &>(*this) >> make_pvp(path, data);
                    return boost::python::object(data);
                }

                boost::python::object load_scalar(std::string const & path) {
                    std::string data;
                    static_cast<alps::hdf5::archive &>(*this) >> make_pvp(path, data);
                    return boost::python::str(data);
                }

                template<typename T> boost::python::object load_numpy(std::string const & path, int type) {
                    std::vector<std::size_t> ext(extent(path));
                    if (type == PyArray_CFLOAT || type == PyArray_CDOUBLE || type == PyArray_CLONGDOUBLE) {
                        if (ext.size() > 1)
                            ext.pop_back();
                        else
                            ext.back() = 1;
                    }
                    std::vector<npy_intp> npextent(ext.begin(), ext.end());
                    std::size_t len = std::accumulate(ext.begin(), ext.end(), std::size_t(1), std::multiplies<std::size_t>());
                    boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(npextent.size(), &npextent.front(), type)));
                    if (len) {
                        boost::scoped_ptr<T> raw(new T[len]);
                        std::pair<T *, std::vector<std::size_t> > data(raw.get(), ext);
                        static_cast<alps::hdf5::archive &>(*this) >> make_pvp(path, data);
                        memcpy(PyArray_DATA(obj.ptr()), raw.get(), PyArray_ITEMSIZE(obj.ptr()) * PyArray_SIZE(obj.ptr()));
                    }
                    return obj;
                }

                template<typename T> void py_save_scalar(std::string const & path, T data) {
                    using alps::make_pvp;
                    static_cast<alps::hdf5::archive &>(*this) << make_pvp(path, data);
                }
                std::pair<bool, std::vector<std::size_t> > py_save_list_extent(boost::python::list const & data) {
                    std::vector<std::string> scalar_types;
                    scalar_types.push_back("int");
                    scalar_types.push_back("long");
                    scalar_types.push_back("float");
                    scalar_types.push_back("complex");
                    scalar_types.push_back("str");
                    scalar_types.push_back("numpy.str");
                    scalar_types.push_back("numpy.bool");
                    scalar_types.push_back("numpy.int8");
                    scalar_types.push_back("numpy.int16");
                    scalar_types.push_back("numpy.int32");
                    scalar_types.push_back("numpy.int64");
                    scalar_types.push_back("numpy.uint8");
                    scalar_types.push_back("numpy.uint16");
                    scalar_types.push_back("numpy.uint32");
                    scalar_types.push_back("numpy.uint64");
                    scalar_types.push_back("numpy.float32");
                    scalar_types.push_back("numpy.float64");
                    scalar_types.push_back("numpy.complex64");
                    scalar_types.push_back("numpy.complex128");

                    boost::python::ssize_t size = boost::python::len(data);
                    std::string first_dtype = boost::python::object(data[0]).ptr()->ob_type->tp_name;
                    bool first_homogenious = true, next_homogenious;
                    std::vector<std::size_t> first_extent, next_extent;
                    if (first_dtype == "list")
                        boost::tie(first_homogenious, first_extent) = py_save_list_extent(boost::python::extract<boost::python::list>(data[0]));
                    else if (first_dtype == "numpy.ndarray") {
                        PyObject* arr = boost::python::object(data[0]).ptr();
                        first_extent = std::vector<std::size_t>(PyArray_DIMS(arr), PyArray_DIMS(arr) + PyArray_NDIM(arr));
                    }
                    if (!first_homogenious)
                        return std::make_pair(false, std::vector<std::size_t>());
                    for(boost::python::ssize_t i = 0; i < size; ++i) {
                        std::string dtype = boost::python::object(data[i]).ptr()->ob_type->tp_name;
                        if (dtype == "list") {
                            boost::tie(next_homogenious, next_extent) = py_save_list_extent(boost::python::extract<boost::python::list>(data[i]));
                            if (!next_homogenious || first_extent.size() != next_extent.size() || !equal(first_extent.begin(), first_extent.end(), next_extent.begin()))
                                return make_pair(false, std::vector<std::size_t>());
                        } else if (dtype == "numpy.ndarray") {
                            PyObject* arr = boost::python::object(data[i]).ptr();
                            next_extent = std::vector<std::size_t>(PyArray_DIMS(arr), PyArray_DIMS(arr) + PyArray_NDIM(arr));
                            if (first_extent.size() != next_extent.size() || !equal(first_extent.begin(), first_extent.end(), next_extent.begin()))
                                return make_pair(false, std::vector<std::size_t>());
                        } else if (first_dtype != dtype || find(scalar_types.begin(), scalar_types.end(), dtype) == scalar_types.end())
                            return std::make_pair(false, std::vector<std::size_t>());
                    }
                    std::vector<std::size_t> extent(1, size);
                    if (first_dtype == "list")
                        copy(first_extent.begin(), first_extent.end(), back_inserter(extent));
                    else if (first_dtype == "numpy.ndarray")
                        copy(first_extent.begin(), first_extent.end(), back_inserter(extent));
                    return std::make_pair(true, extent);
                }
                void py_save_list_save(
                      std::string const & path
                    , boost::python::list data
                    , std::vector<std::size_t> size = std::vector<std::size_t>()
                    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                    , std::vector<std::size_t> offset = std::vector<std::size_t>()
                ) {
                    size.push_back(boost::python::len(data));
                    chunk.push_back(1);
                    offset.push_back(0);
                    alps::hdf5::archive & ar = static_cast<alps::hdf5::archive &>(*this);
                    for(boost::python::ssize_t i = 0; i < boost::python::len(data); ++i) {
                        offset.back() = i;
                        boost::python::object item = data[i];
                        std::string dtype = item.ptr()->ob_type->tp_name;
                        if (dtype == "list") py_save_list_save(path, boost::python::extract<boost::python::list>(item)(), size, chunk, offset);
                        else if (dtype == "int") save(ar, path, boost::python::extract<int>(item)(), size, chunk, offset);
                        else if (dtype == "long") save(ar, path, boost::python::extract<long>(item)(), size, chunk, offset);
                        else if (dtype == "float") save(ar, path, boost::python::extract<double>(item)(), size, chunk, offset);
                        else if (dtype == "complex") save(ar, path, boost::python::extract<std::complex<double> >(item)(), size, chunk, offset);
                        else if (dtype == "str") save(ar, path, boost::python::extract<std::string>(item)(), size, chunk, offset);
                        else if (dtype == "numpy.ndarray") py_save_numpy(path, boost::python::extract<boost::python::numeric::array>(item)(), size, chunk, offset);
                        else if (dtype == "numpy.str") save(ar, path, boost::python::call_method<std::string>(item.ptr(), "__str__"), size, chunk, offset);
                        else if (dtype == "numpy.bool") save(ar, path, boost::python::call_method<bool>(item.ptr(), "__bool__"), size, chunk, offset);
                        else if (dtype == "numpy.int8") save(ar, path, static_cast<boost::int8_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.int16") save(ar, path, static_cast<boost::int16_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.int32") save(ar, path, static_cast<boost::int32_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.int64") save(ar, path, static_cast<boost::int64_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.uint8") save(ar, path, static_cast<boost::uint8_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.uint16") save(ar, path, static_cast<boost::uint16_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.uint32") save(ar, path, static_cast<boost::uint32_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.uint64") save(ar, path, static_cast<boost::uint64_t>(boost::python::call_method<long>(item.ptr(), "__long__")), size, chunk, offset);
                        else if (dtype == "numpy.float32") save(ar, path, static_cast<float>(boost::python::call_method<double>(item.ptr(), "__float__")), size, chunk, offset);
                        else if (dtype == "numpy.float64") save(ar, path, static_cast<double>(boost::python::call_method<double>(item.ptr(), "__float__")), size, chunk, offset);
                        else if (dtype == "numpy.complex64") save(ar, path, std::complex<float>(
                              boost::python::call_method<double>(PyObject_GetAttr(item.ptr(), boost::python::str("real").ptr()), "__float__")
                            , boost::python::call_method<double>(PyObject_GetAttr(item.ptr(), boost::python::str("imag").ptr()), "__float__")
                        ));
                        else if (dtype == "numpy.complex128") save(ar, path, std::complex<double>(
                              boost::python::call_method<double>(PyObject_GetAttr(item.ptr(), boost::python::str("real").ptr()), "__float__")
                            , boost::python::call_method<double>(PyObject_GetAttr(item.ptr(), boost::python::str("imag").ptr()), "__float__")
                        ));
                        else
                            throw std::runtime_error("Unsupported type: " + dtype + ALPS_STACKTRACE);
                    }
                }
                void py_save_list(std::string const & path, boost::python::list const & data) {
                    boost::python::ssize_t size = boost::python::len(data);
                    if (size == 0)
                        write(path, static_cast<int const *>(NULL), std::vector<std::size_t>());
                    bool homogenious  = py_save_list_extent(data).first;
                    if (is_group(path))
                        delete_group(path);
                    if (homogenious)
                        py_save_list_save(path, data);
                    else {
                        if (is_data(path))
                            delete_data(path);
                        for(boost::python::ssize_t i = 0; i < size; ++i)
                            py_save(path + "/" + cast<std::string>(i), data[i]);
                    
                    }
                }
                void py_save_dict(std::string const & path, boost::python::dict const & data) {
                    const boost::python::object kit = data.iterkeys();
                    const boost::python::object vit = data.itervalues();
                    std::size_t size = boost::python::len(data);
                    for (std::size_t i = 0; i < size; ++i)
                        py_save(
                              path + "/" + encode_segment(boost::python::call_method<std::string>(kit.attr("next")().ptr(), "__str__"))
                            , vit.attr("next")()
                        );
                }
                void py_save_numpy(
                      std::string const & path
                    , boost::python::numeric::array data
                    , std::vector<std::size_t> size = std::vector<std::size_t>()
                    , std::vector<std::size_t> chunk = std::vector<std::size_t>()
                    , std::vector<std::size_t> offset = std::vector<std::size_t>()
                ) {
                    if (!PyArray_Check(data.ptr()))
                        throw std::runtime_error("invalid numpy data" + ALPS_STACKTRACE);
                    else if (!PyArray_ISCONTIGUOUS(data.ptr()))
                        throw std::runtime_error("numpy array is not continous" + ALPS_STACKTRACE);
                    else if (!PyArray_ISNOTSWAPPED(data.ptr()))
                        throw std::runtime_error("numpy array is not native" + ALPS_STACKTRACE);
                    std::vector<std::size_t> extent(PyArray_DIMS(data.ptr()), PyArray_DIMS(data.ptr()) + PyArray_NDIM(data.ptr()));
                    std::copy(extent.begin(), extent.end(), std::back_inserter(size));
                    std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
                    std::fill_n(std::back_inserter(offset), extent.size(), 0);
                    alps::hdf5::archive & ar = static_cast<alps::hdf5::archive &>(*this);
                    if (false);
                    #define NGS_PYTHON_HDF5_CHECK_NUMPY(T, N)                                                                                       \
                        else if (PyArray_DESCR(data.ptr())->type_num == N)                                                                          \
                            save(ar, path, *static_cast< T const *>(PyArray_DATA(data.ptr())), size, chunk, offset);
                    #define NGS_PYTHON_HDF5_CHECK_NUMPY_CPLX(T, N)                                                                                  \
                        else if (PyArray_DESCR(data.ptr())->type_num == N) {                                                                        \
                            save(ar, path, *static_cast< T const *>(PyArray_DATA(data.ptr())), size, chunk, offset);                                \
                            ar.set_complex(path);                                                                                                   \
                        }
                    NGS_PYTHON_HDF5_CHECK_NUMPY(bool,PyArray_BOOL)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(char, PyArray_CHAR)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(unsigned char, PyArray_UBYTE)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(signed char, PyArray_BYTE)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(short, PyArray_SHORT)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(unsigned short, PyArray_USHORT)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(int, PyArray_INT)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(unsigned int, PyArray_UINT)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(long, PyArray_LONG)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(long long, PyArray_LONGLONG)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(unsigned long long, PyArray_ULONGLONG)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(float, PyArray_FLOAT)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(double, PyArray_DOUBLE)
                    NGS_PYTHON_HDF5_CHECK_NUMPY(long double, PyArray_LONGDOUBLE)
                    NGS_PYTHON_HDF5_CHECK_NUMPY_CPLX(std::complex<float>, PyArray_CFLOAT)
                    NGS_PYTHON_HDF5_CHECK_NUMPY_CPLX(std::complex<double>,PyArray_CDOUBLE)
                    NGS_PYTHON_HDF5_CHECK_NUMPY_CPLX(std::complex<long double>, PyArray_CLONGDOUBLE)
                    #undef NGS_PYTHON_HDF5_CHECK_NUMPY
                    #undef NGS_PYTHON_HDF5_CHECK_NUMPY_CPLX
                    else
                        throw std::runtime_error("unknown numpy element type" + ALPS_STACKTRACE);
                }
         };
    }
}

BOOST_PYTHON_MODULE(pyngshdf5_c) {

    // TODO: move to ownl cpp file and include everywhere
    boost::python::to_python_converter<
      std::string,
      alps::detail::std_string_to_python
    >();

    boost::python::to_python_converter<
        std::vector<std::string>,
        alps::detail::std_vector_string_to_python
    >();

    boost::python::class_<alps::detail::hdf5_archive_export, boost::python::bases<alps::hdf5::archive> >(
          "hdf5_archive_impl",
          boost::python::init<std::string, std::string>()
    )
        .def("__deepcopy__", &alps::python::make_copy<alps::detail::hdf5_archive_export>)
        .add_property("filename", &alps::detail::hdf5_archive_export::py_filename)
        .add_property("context", &alps::detail::hdf5_archive_export::py_context)
        .def("set_context", &alps::detail::hdf5_archive_export::set_context)
        .def("is_group", &alps::detail::hdf5_archive_export::is_group)
        .def("is_data", &alps::detail::hdf5_archive_export::is_data)
        .def("is_attribute", &alps::detail::hdf5_archive_export::is_attribute)
        .def("extent", &alps::detail::hdf5_archive_export::py_extent)
        .def("dimensions", &alps::detail::hdf5_archive_export::dimensions)
        .def("is_scalar", &alps::detail::hdf5_archive_export::is_scalar)
        .def("is_complex", &alps::detail::hdf5_archive_export::is_complex)
        .def("is_null", &alps::detail::hdf5_archive_export::is_null)
        .def("list_children", &alps::detail::hdf5_archive_export::list_children)
        .def("list_attributes", &alps::detail::hdf5_archive_export::list_attributes)
        .def("save", &alps::detail::hdf5_archive_export::py_save)
        .def("load", &alps::detail::hdf5_archive_export::py_load)
        .def("create_group", &alps::detail::hdf5_archive_export::create_group)
        .def("delete_data", &alps::detail::hdf5_archive_export::delete_data)
        .def("delete_group", &alps::detail::hdf5_archive_export::delete_group)
        .def("delete_attribute", &alps::detail::hdf5_archive_export::delete_attribute)
    ;
}
