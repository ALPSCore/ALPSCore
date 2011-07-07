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

#define PY_ARRAY_UNIQUE_SYMBOL pyngshdf5_PyArrayHandle

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/macros.hpp>
#include <alps/ngs/hdf5/pair.hpp>
#include <alps/ngs/hdf5/vector.hpp>
#include <alps/ngs/hdf5/complex.hpp>
#include <alps/ngs/boost_python.hpp>

#include <alps/python/make_copy.hpp>

#include <boost/scoped_ptr.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/to_python_converter.hpp>

#include <numpy/arrayobject.h>


namespace alps {
    namespace detail {

        void import_numpy() {
            static bool inited = false;
            if (!inited) {
                import_array();  
                boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
                inited = true;
            }
        }

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

        class hdf5_archive_export : public alps::hdf5::archive {
            using alps::hdf5::archive::extent;
            using alps::hdf5::archive::is_datatype;

            public:
            
                hdf5_archive_export(std::string const & filename, std::size_t props)
                    : alps::hdf5::archive(filename, props)
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

                void py_save(std::string const & path, boost::python::object const & data, std::string const & dtype) {
                    using alps::make_pvp;
                    import_numpy();
                    if (false);
                    #define NGS_PYTHON_HDF5_CHECK_SCALAR(N, T)                                                                                              \
                        else if (dtype == N)                                                                                                                \
                            static_cast<alps::hdf5::archive &>(*this) << make_pvp(path, boost::python::extract< T >(data)());
                    NGS_PYTHON_HDF5_CHECK_SCALAR("bool", bool)
                    NGS_PYTHON_HDF5_CHECK_SCALAR("int", int)
                    NGS_PYTHON_HDF5_CHECK_SCALAR("long", long)
                    NGS_PYTHON_HDF5_CHECK_SCALAR("float", double)
                    NGS_PYTHON_HDF5_CHECK_SCALAR("complex", std::complex<double>)
                    NGS_PYTHON_HDF5_CHECK_SCALAR("str", std::string)
                    #undef NGS_PYTHON_HDF5_CHECK_SCALAR
                    #define NGS_PYTHON_HDF5_CHECK_NUMPY(N, T)                                                                                               \
                        else if (dtype == N) {                                                                                                              \
                            if (PyArray_Check(data.ptr())) {                                                                                                \
                                if (!PyArray_ISCONTIGUOUS(data.ptr()))                                                                                      \
                                    ALPS_NGS_THROW_RUNTIME_ERROR("numpy array is not continous");                                                           \
                                else if (!PyArray_ISNOTSWAPPED(data.ptr()))                                                                                 \
                                    ALPS_NGS_THROW_RUNTIME_ERROR("numpy array is not native");                                                              \
                                static_cast<alps::hdf5::archive &>(*this) << make_pvp(                                                                      \
                                      path                                                                                                                  \
                                    , std::make_pair(                                                                                                       \
                                          static_cast< T const *>(PyArray_DATA(data.ptr()))                                                                 \
                                        , std::vector<std::size_t>(PyArray_DIMS(data.ptr()), PyArray_DIMS(data.ptr()) + PyArray_NDIM(data.ptr()))           \
                                    )                                                                                                                       \
                                );                                                                                                                          \
                            } else                                                                                                                          \
                                static_cast<alps::hdf5::archive &>(*this) << make_pvp(path, boost::python::extract< T >(data)());                           \
                        }
                    NGS_PYTHON_HDF5_CHECK_NUMPY("bool", bool)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("string_", char)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("int8", short)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("uint8", unsigned short)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("int16", short)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("uint16", unsigned short)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("int32", int)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("uint32", unsigned int)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("float32", float)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("complex32", std::complex<float>)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("int", int)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("int64", long)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("uint64", unsigned long)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("float64", double)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("float", double)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("float128", long double)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("complex128", std::complex<long double>)
                    NGS_PYTHON_HDF5_CHECK_NUMPY("complex", std::complex<long double>)
                    #undef NGS_PYTHON_HDF5_CHECK_NUMPY
                    else
                        ALPS_NGS_THROW_RUNTIME_ERROR("Unsupported type");
                }

                boost::python::object py_load(std::string const & path) {
                    import_numpy();
                    if (is_scalar(path) || (is_datatype<double>(path) && is_complex(path) && extent(path).size() == 1 && extent(path)[0] == 2)) {
                        if (is_datatype<std::string>(path))
                            return load_scalar<std::string>(path);
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
                            ALPS_NGS_THROW_RUNTIME_ERROR("Unsupported type.");
                    } else if (is_datatype<std::string>(path)) {
                        if (dimensions(path) != 1)
                            ALPS_NGS_THROW_RUNTIME_ERROR("More than 1 Dimension is not supported.");
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
                        ALPS_NGS_THROW_RUNTIME_ERROR("Unsupported type.")
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
          boost::python::init<std::string, std::size_t>()
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
