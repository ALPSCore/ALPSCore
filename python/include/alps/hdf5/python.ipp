/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
 
#include <alps/hdf5/python.hpp>

#include <alps/utilities/type_wrapper.hpp>
#include <alps/utilities/import_numpy.hpp>
#include <alps/utilities/get_numpy_type.hpp>
#include <alps/utilities/extract_from_pyobject.hpp>

#include <boost/python/numeric.hpp>
#include <numpy/arrayobject.h>

namespace alps {
    namespace hdf5 {

        namespace detail {

            template<typename T> bool is_vectorizable_generic(T const & value) {
                static char const * scalar_types[] = { "int", "long", "float", "complex", "str"
                    , "numpy.str", "numpy.bool", "numpy.int8", "numpy.int16", "numpy.int32", "numpy.int64", "numpy.uint8"
                    , "numpy.uint16", "numpy.uint32", "numpy.uint64", "numpy.float32", "numpy.float64", "numpy.complex64", "numpy.complex128" };
                using boost::python::len;
                using alps::hdf5::get_extent;
                boost::python::ssize_t size = len(value);
                if (size == 0)
                    return true;
                else {
                    std::string first_dtype = boost::python::object(value[0]).ptr()->ob_type->tp_name;
                    bool next_homogenious;
                    std::vector<std::size_t> first_extent;
                    if (first_dtype == "list") {
                        if (!is_vectorizable<boost::python::list>::apply(boost::python::extract<boost::python::list>(value[0])()))
                            return false;
                        first_extent = get_extent(boost::python::extract<boost::python::list>(value[0])());
                    } else if (first_dtype == "tuple") {
                        if (!is_vectorizable<boost::python::tuple>::apply(boost::python::extract<boost::python::tuple>(value[0])()))
                            return false;
                        first_extent = get_extent(boost::python::extract<boost::python::tuple>(value[0])());
                    } else if (first_dtype == "numpy.ndarray")
                        first_extent = get_extent(boost::python::extract<boost::python::numeric::array>(value[0])());
                    for(boost::python::ssize_t i = 0; i < size; ++i) {
                        std::string dtype = boost::python::object(value[i]).ptr()->ob_type->tp_name;
                        if (dtype == "list") {
                            if (!is_vectorizable<boost::python::list>::apply(boost::python::extract<boost::python::list>(value[i])()))
                                return false;
                            std::vector<std::size_t> extent = get_extent(boost::python::extract<boost::python::list>(value[i])());
                            if (first_extent.size() != extent.size() || !std::equal(first_extent.begin(), first_extent.end(), extent.begin()))
                                return false;
                        } else if (dtype == "tuple") {
                            if (!is_vectorizable<boost::python::tuple>::apply(boost::python::extract<boost::python::tuple>(value[i])()))
                                return false;
                            std::vector<std::size_t> extent = get_extent(boost::python::extract<boost::python::tuple>(value[i])());
                            if (first_extent.size() != extent.size() || !std::equal(first_extent.begin(), first_extent.end(), extent.begin()))
                                return false;
                        } else if (dtype == "numpy.ndarray") {
                            std::vector<std::size_t> extent = get_extent(boost::python::extract<boost::python::numeric::array>(value[i])());
                            if (first_extent.size() != extent.size() || !std::equal(first_extent.begin(), first_extent.end(), extent.begin()))
                                return false;
                        } else if (first_dtype != dtype || find(scalar_types, scalar_types + 19, dtype) == scalar_types + 19)
                            return false;
                    }
                    return true;
                }
            }
            bool is_vectorizable<boost::python::list>::apply(boost::python::list const & value) {
                return is_vectorizable_generic<boost::python::list>(value);
            }
            bool is_vectorizable<boost::python::tuple>::apply(boost::python::tuple const & value) {
                return is_vectorizable_generic<boost::python::tuple>(value);
            }

            template<typename T> std::vector<std::size_t> get_extent_generic(T const & value) {
                using boost::python::len;
                using alps::hdf5::get_extent;
                using alps::hdf5::is_vectorizable;
                if (!is_vectorizable(value))
                    throw archive_error("no rectengual matrix" + ALPS_STACKTRACE);
                std::vector<std::size_t> extent(1, len(value));
                std::string first_dtype = boost::python::object(value[0]).ptr()->ob_type->tp_name;
                if (first_dtype == "list") {
                    std::vector<std::size_t> first_extent(get_extent(boost::python::extract<boost::python::list>(value[0])()));
                    copy(first_extent.begin(), first_extent.end(), back_inserter(extent));
                } else if (first_dtype == "tuple") {
                    std::vector<std::size_t> first_extent(get_extent(boost::python::extract<boost::python::tuple>(value[0])()));
                    copy(first_extent.begin(), first_extent.end(), back_inserter(extent));
                } else if (first_dtype == "numpy.ndarray") {
                    std::vector<std::size_t> first_extent = get_extent(boost::python::extract<boost::python::numeric::array>(value[0])());
                    copy(first_extent.begin(), first_extent.end(), back_inserter(extent));
                }
                return extent;
            }
            std::vector<std::size_t> get_extent<boost::python::list>::apply(boost::python::list const & value) {
                return get_extent_generic<boost::python::list>(value);
            }
            std::vector<std::size_t> get_extent<boost::python::tuple>::apply(boost::python::tuple const & value) {
                return get_extent_generic<boost::python::tuple>(value);
            }

            void set_extent<boost::python::list>::apply(boost::python::list & value, std::vector<std::size_t> const & extent) {}
        }

        template<typename T> void save_generic(
              archive & ar
            , std::string const & path
            , T const & value
            , std::vector<std::size_t> size
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            using alps::cast;
            using boost::python::len;
            if (ar.is_group(path))
                ar.delete_group(path);
            if (len(value) == 0)
                ar.write(path, static_cast<int const *>(NULL), std::vector<std::size_t>());
            else if (is_vectorizable(value)) {
                size.push_back(len(value));
                chunk.push_back(1);
                offset.push_back(0);
                for(boost::python::ssize_t i = 0; i < len(value); ++i) {
                    offset.back() = i;
                    save(ar, path, boost::python::object(value[i]), size, chunk, offset);
                }
            } else {
                if (ar.is_data(path))
                    ar.delete_data(path);
                for(boost::python::ssize_t i = 0; i < len(value); ++i)
                    save(ar, path + "/" + cast<std::string>(i), boost::python::object(value[i]));
            }
        }

        void save(
              archive & ar
            , std::string const & path
            , boost::python::list const & value
            , std::vector<std::size_t> size
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            save_generic<boost::python::list>(ar, path, value, size, chunk, offset);
        }

        void save(
              archive & ar
            , std::string const & path
            , boost::python::tuple const & value
            , std::vector<std::size_t> size
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            save_generic<boost::python::tuple>(ar, path, value, size, chunk, offset);
        }        

        void load(
              archive & ar
            , std::string const & path
            , boost::python::list & value
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            if (ar.is_group(path)) {
                std::vector<std::string> list = ar.list_children(path);
                if (list.size()) {
                    std::vector<boost::python::object> data;
                    load(ar, path, data, chunk, offset);
                    for (std::vector<boost::python::object>::const_iterator it = data.begin(); it != data.end(); ++it)
                         value.append(*it);
                }
            } else if (!ar.is_scalar(path) && ar.is_datatype<std::string>(path)) {
                if (ar.dimensions(path) != 1)
                    throw archive_error("More than 1 Dimension is not supported." + ALPS_STACKTRACE);
                std::vector<std::string> data;
                load(ar, path, data, chunk, offset);
                for (std::vector<std::string>::const_iterator it = data.begin(); it != data.end(); ++it)
                     value.append(boost::python::str(*it));
            }
        }

        namespace detail {

            bool is_vectorizable<boost::python::numeric::array>::apply(boost::python::numeric::array const & value) {
                return true;
            }

            std::vector<std::size_t> get_extent<boost::python::numeric::array>::apply(boost::python::numeric::array const & value) {
                if (!is_vectorizable<boost::python::numeric::array>::apply(value))
                    throw archive_error("no rectengual matrix" + ALPS_STACKTRACE);
                return std::vector<std::size_t>(PyArray_DIMS(value.ptr()), PyArray_DIMS(value.ptr()) + PyArray_NDIM(value.ptr()));
            }

            // To set the extent of a numpy array, we need the type, extent is set in load
            void set_extent<boost::python::numeric::array>::apply(boost::python::numeric::array & value, std::vector<std::size_t> const & extent) {}

            template <typename T> void load_python_numeric(
                  archive & ar
                , std::string const & path
                , boost::python::numeric::array & value
                , std::vector<std::size_t> chunk
                , std::vector<std::size_t> offset
                , int type
            ) {
                std::vector<std::size_t> extent(ar.extent(path));
                if (ar.is_complex(path))
                    extent.pop_back();
                std::vector<npy_intp> npextent(extent.begin(), extent.end());
                std::size_t len = std::accumulate(extent.begin(), extent.end(), std::size_t(1), std::multiplies<std::size_t>());
                value = boost::python::numeric::array(boost::python::handle<>(PyArray_SimpleNew(npextent.size(), &npextent.front(), type)));
                if (len) {
                    boost::scoped_ptr<T> raw(new T[len]);
                    std::pair<T *, std::vector<std::size_t> > data(raw.get(), extent);
                    load(ar, path, data, chunk, offset);
                    memcpy(PyArray_DATA(value.ptr()), raw.get(), PyArray_ITEMSIZE(value.ptr()) * PyArray_SIZE(value.ptr()));
                }
            }
        }

        void save(
              archive & ar
            , std::string const & path
            , boost::python::numeric::array const & value
            , std::vector<std::size_t> size
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            using ::alps::detail::import_numpy;
            import_numpy();
            if (ar.is_group(path))
                ar.delete_group(path);
            PyArrayObject * ptr = (PyArrayObject *)value.ptr();
            if (!PyArray_Check(ptr))
                throw std::runtime_error("invalid numpy data" + ALPS_STACKTRACE);
            else if (!PyArray_ISNOTSWAPPED(ptr))
                throw std::runtime_error("numpy array is not native" + ALPS_STACKTRACE);
            else if (!(ptr = PyArray_GETCONTIGUOUS(ptr)))
                throw std::runtime_error("numpy array cannot be converted to continous array" + ALPS_STACKTRACE);
            std::vector<std::size_t> extent(PyArray_DIMS(ptr), PyArray_DIMS(ptr) + PyArray_NDIM(ptr));
            std::copy(extent.begin(), extent.end(), std::back_inserter(size));
            std::copy(extent.begin(), extent.end(), std::back_inserter(chunk));
            std::fill_n(std::back_inserter(offset), extent.size(), 0);
            if (false)
                ;
            #define ALPS_PYTHON_HDF5_CHECK_NUMPY(T)                                                                                         \
                else if (PyArray_DESCR(ptr)->type_num == ::alps::detail::get_numpy_type(alps::detail::type_wrapper< T >::type())) {         \
                    save(ar, path, *static_cast< T const *>(PyArray_DATA(ptr)), size, chunk, offset);                                       \
                    if (has_complex_elements< T >::value)                                                                                   \
                        ar.set_complex(path);                                                                                               \
                }
            ALPS_FOREACH_NATIVE_NUMPY_TYPE(ALPS_PYTHON_HDF5_CHECK_NUMPY)
            #undef ALPS_PYTHON_HDF5_CHECK_NUMPY
            else
                throw std::runtime_error("unknown numpy element type" + ALPS_STACKTRACE);
            Py_DECREF((PyObject *)ptr);
        }

        void load(
              archive & ar
            , std::string const & path
            , boost::python::numeric::array & value
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            using ::alps::detail::import_numpy;
            import_numpy();
            if (false);
            #define ALPS_PYTHON_HDF5_LOAD_NUMPY(T)                                                                                                              \
                else if (ar.is_datatype<scalar_type< T >::type>(path) && ar.is_complex(path) == has_complex_elements< T >::value)                               \
                    detail::load_python_numeric< T >(ar, path, value, chunk, offset, ::alps::detail::get_numpy_type(alps::detail::type_wrapper< T >::type()));
            ALPS_FOREACH_NATIVE_NUMPY_TYPE(ALPS_PYTHON_HDF5_LOAD_NUMPY)
            #undef ALPS_PYTHON_HDF5_LOAD_NUMPY
            else
                throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
        }
        
        void save(
              archive & ar
            , std::string const & path
            , boost::python::dict const & value
            , std::vector<std::size_t> size
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            if (ar.is_group(path))
                ar.delete_group(path);
            const boost::python::object kit = value.iterkeys();
            const boost::python::object vit = value.itervalues();
            using boost::python::len;
            for (boost::python::ssize_t i = 0; i < len(value); ++i)
                save(
                      ar
                    , ar.complete_path(path) + "/" + ar.encode_segment(boost::python::call_method<std::string>(kit.attr("next")().ptr(), "__str__"))
                    , vit.attr("next")()
                );
        }

        void load(
              archive & ar
            , std::string const & path
            , boost::python::dict & value
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            std::vector<std::string> children = ar.list_children(path);
            for (std::vector<std::string>::const_iterator it = children.begin(); it != children.end(); ++it) {
                boost::python::object item;
                load(ar, path + "/" + *it, item);
                boost::python::call_method<void>(value.ptr(), "__setitem__", *it, item);
            }
        }

        namespace detail {

            bool is_vectorizable<boost::python::object>::apply(boost::python::object const & value) {
                static char const * scalar_types[] = { "int", "long", "float", "complex", "str"
                    , "numpy.str", "numpy.bool", "numpy.int8", "numpy.int16", "numpy.int32", "numpy.int64", "numpy.uint8"
                    , "numpy.uint16", "numpy.uint32", "numpy.uint64", "numpy.float32", "numpy.float64", "numpy.complex64", "numpy.complex128" };
                std::string dtype = value.ptr()->ob_type->tp_name;
                if (dtype == "list")
                    return is_vectorizable<boost::python::list>::apply(boost::python::extract<boost::python::list>(value)());
                else if (dtype == "numpy.ndarray")
                    return is_vectorizable<boost::python::numeric::array>::apply(boost::python::extract<boost::python::numeric::array>(value)());
                return find(scalar_types, scalar_types + 19, dtype) < scalar_types + 19;
            }

            std::vector<std::size_t> get_extent<boost::python::object>::apply(boost::python::object const & value) {
                using alps::hdf5::get_extent;
                std::string dtype = value.ptr()->ob_type->tp_name;
                if (!is_vectorizable<boost::python::object>::apply(value))
                    throw archive_error("no rectengual matrix" + ALPS_STACKTRACE);
                if (dtype == "list")
                    return get_extent(boost::python::extract<boost::python::list>(value)());
                else if (dtype == "numpy.ndarray")
                    return get_extent(boost::python::extract<boost::python::numeric::array>(value)());
                else
                    return std::vector<std::size_t>();
            }

            void set_extent<boost::python::object>::apply(boost::python::object & value, std::vector<std::size_t> const & extent) {}

            struct save_python_object_visitor {
                save_python_object_visitor(
                      archive & ar
                    , std::string const & path
                    , std::vector<std::size_t> size
                    , std::vector<std::size_t> chunk
                    , std::vector<std::size_t> offset
                )
                    : _ar(ar)
                    , _path(path)
                    , _size(size)
                    , _chunk(chunk)
                    , _offset(offset)
                {}
                template<typename T> void operator()(T const & value) {
                    save(_ar, _path, value, _size, _chunk, _offset);
                    if (has_complex_elements< T >::value)
                        _ar.set_complex(_path);
                }
                template<typename T> void operator()(T const *, std::vector<std::size_t>) {
                    throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
                }
                archive & _ar;
                std::string const & _path;
                std::vector<std::size_t> _size;
                std::vector<std::size_t> _chunk;
                std::vector<std::size_t> _offset;
            };

            template <typename T> void load_python_object(
                  archive & ar
                , std::string const & path
                , boost::python::object & value
                , std::vector<std::size_t> chunk
                , std::vector<std::size_t> offset
                , int type
            ) {
                T data;
                load(ar, path, data, chunk, offset);
                value = boost::python::object(data);
            }
        }

        void save(
              archive & ar
            , std::string const & path
            , boost::python::object const & value
            , std::vector<std::size_t> size
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            std::string dtype = value.ptr()->ob_type->tp_name;
            if (dtype == "numpy.ndarray")
                save(ar, path, boost::python::extract<boost::python::numeric::array>(value)(), size, chunk, offset);
            else if (PyObject_HasAttrString(value.ptr(), "save") && std::string(PyObject_GetAttrString(value.ptr(), "save")->ob_type->tp_name) == "instancemethod") {
                std::string context = ar.get_context();
                ar.set_context(ar.complete_path(path));
                boost::python::call_method<void>(value.ptr(), "save", boost::python::object(ar));
                ar.set_context(context);
            } else {
                using ::alps::detail::extract_from_pyobject;
                detail::save_python_object_visitor visitor(ar, path, size, chunk, offset);
                extract_from_pyobject(visitor, value);
            }
        }

        void load(
              archive & ar
            , std::string const & path
            , boost::python::object & value
            , std::vector<std::size_t> chunk
            , std::vector<std::size_t> offset
        ) {
            if (PyObject_HasAttrString(value.ptr(), "load") && std::string(PyObject_GetAttrString(value.ptr(), "load")->ob_type->tp_name) == "MethodType") {
                std::string context = ar.get_context();
                ar.set_context(ar.complete_path(path));
                boost::python::call_method<void>(value.ptr(), "load", boost::python::object(ar), path);
                ar.set_context(context);
            } else if (ar.is_group(path)) {
                std::vector<std::string> list = ar.list_children(path);
                bool is_list = list.size();
                for (std::vector<std::string>::const_iterator it = list.begin(); is_list && it != list.end(); ++it) {
                    for (std::string::const_iterator jt = it->begin(); is_list && jt != it->end(); ++jt)
                        if (std::string("1234567890").find_first_of(*jt) == std::string::npos)
                            is_list = false;
                    if (is_list && alps::cast<unsigned>(*it) > list.size() - 1)
                        is_list = false;
                }
                if (is_list) {
                    value = boost::python::list();
                    load(ar, path, static_cast<boost::python::list &>(value), chunk, offset);
                } else {
                    value = boost::python::dict();
                    load(ar, path, static_cast<boost::python::dict &>(value), chunk, offset);
                }
            } else if (ar.is_scalar(path) || (ar.is_datatype<double>(path) && ar.is_complex(path) && ar.extent(path).size() == 1 && ar.extent(path)[0] == 2)) {
                if (ar.is_datatype<std::string>(path)) {
                    std::string data;
                    load(ar, path, data, chunk, offset);
                    value = boost::python::str(data);
                #define ALPS_PYTHON_HDF5_LOAD_SCALAR_NUMPY(T)                                                                                                   \
                } else if (ar.is_datatype<scalar_type< T >::type>(path) && ar.is_complex(path) == has_complex_elements< T >::value) {                           \
                    detail::load_python_object< T >(ar, path, value, chunk, offset, ::alps::detail::get_numpy_type(alps::detail::type_wrapper< T >::type()));
                ALPS_FOREACH_NATIVE_NUMPY_TYPE(ALPS_PYTHON_HDF5_LOAD_SCALAR_NUMPY)
                #undef ALPS_PYTHON_HDF5_LOAD_SCALAR_NUMPY
                } else
                    throw std::runtime_error("Unsupported type." + ALPS_STACKTRACE);
            } else if (ar.is_datatype<std::string>(path)) {
                value = boost::python::list();
                load(ar, path, static_cast<boost::python::list &>(value), chunk, offset);
            } else {
                boost::python::numeric::array array(boost::python::make_tuple());
                load(ar, path, array, chunk, offset);
                value = array;
            }
        }

    }
}
