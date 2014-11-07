/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#define PY_ARRAY_UNIQUE_SYMBOL pyhdf5_PyArrayHandle

// this must be first
#include <alps/utilities/boost_python.hpp>

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/pair.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/python.hpp>
#include <alps/hdf5/complex.hpp>

#include <alps/utilities/stacktrace.hpp>
#include <alps/utilities/make_copy.hpp>

#include <boost/scoped_ptr.hpp>
#include <boost/python/list.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/numeric.hpp>
#include <boost/python/to_python_converter.hpp>
#include <boost/array.hpp>

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

        boost::python::str python_hdf5_get_filename(alps::hdf5::archive & ar) {
            return boost::python::str(ar.get_filename());
        }

        void python_hdf5_save(alps::hdf5::archive & ar, std::string const & path, boost::python::object const & data) {
            import_numpy();
            ar[path] << data;
        }

        boost::python::object python_hdf5_load(alps::hdf5::archive & ar, std::string const & path) {
            import_numpy();
            boost::python::object value;
            ar[path] >> value;
            return value;
        }
        
        boost::python::list python_hdf5_extent(alps::hdf5::archive & ar, std::string const & path) {
            boost::python::list result;
            std::vector<std::size_t> ext = ar.extent(path);
            if (ar.is_complex(path)) {
                if (ext.size() > 1)
                    ext.pop_back();
                else
                    ext.back() = 1;
            }
            for (std::vector<std::size_t>::const_iterator it = ext.begin(); it != ext.end(); ++it)
                result.append(*it);
            return result;
        }
    
        boost::array<PyObject *, 6> exception_type;
    
        #define TRANSLATE_CPP_ERROR_TO_PYTHON(T, ID)                                                            \
        void translate_ ## T (hdf5:: T const & e) {                                                             \
            std::string message = std::string(e.what()).substr(0, std::string(e.what()).find_first_of('\n'));   \
            PyErr_SetString(exception_type[ID], const_cast<char *>(message.c_str()));                           \
        }
        TRANSLATE_CPP_ERROR_TO_PYTHON(archive_error, 0)
        TRANSLATE_CPP_ERROR_TO_PYTHON(archive_not_found, 1)
        TRANSLATE_CPP_ERROR_TO_PYTHON(archive_closed, 2)
        TRANSLATE_CPP_ERROR_TO_PYTHON(invalid_path, 3)
        TRANSLATE_CPP_ERROR_TO_PYTHON(path_not_found, 4)
        TRANSLATE_CPP_ERROR_TO_PYTHON(wrong_type, 5)

        void register_exception_type(int id, boost::python::object type) {
            Py_INCREF(type.ptr());
            exception_type[id] = type.ptr();
        }
    }
}

BOOST_PYTHON_MODULE(pyhdf5_c) {

    // TODO: move to ownl cpp file and include everywhere
    boost::python::to_python_converter<
      std::string,
      alps::detail::std_string_to_python
    >();

    boost::python::to_python_converter<
        std::vector<std::string>,
        alps::detail::std_vector_string_to_python
    >();

    boost::python::register_exception_translator<alps::hdf5::archive_error>(&alps::detail::translate_archive_error);
    boost::python::register_exception_translator<alps::hdf5::archive_not_found>(&alps::detail::translate_archive_not_found);
    boost::python::register_exception_translator<alps::hdf5::archive_closed>(&alps::detail::translate_archive_closed);
    boost::python::register_exception_translator<alps::hdf5::invalid_path>(&alps::detail::translate_invalid_path);
    boost::python::register_exception_translator<alps::hdf5::path_not_found>(&alps::detail::translate_path_not_found);
    boost::python::register_exception_translator<alps::hdf5::wrong_type>(&alps::detail::translate_wrong_type);

    boost::python::def("register_archive_exception_type", &alps::detail::register_exception_type);    

    boost::python::class_<alps::hdf5::archive>(
          "hdf5_archive_impl",
          boost::python::init<std::string, std::string>()
    )
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::archive>)
        .add_property("filename", &alps::detail::python_hdf5_get_filename)
        .add_property("context", &alps::hdf5::archive::get_context)
        .add_property("is_open", &alps::hdf5::archive::is_open)
        .def("set_context", &alps::hdf5::archive::set_context)
        .def("is_group", &alps::hdf5::archive::is_group)
        .def("is_data", &alps::hdf5::archive::is_data)
        .def("is_attribute", &alps::hdf5::archive::is_attribute)
        .def("is_open", &alps::hdf5::archive::is_open)
        .def("close", &alps::hdf5::archive::close)
        .def("extent", &alps::detail::python_hdf5_extent)
        .def("dimensions", &alps::hdf5::archive::dimensions)
        .def("is_scalar", &alps::hdf5::archive::is_scalar)
        .def("is_complex", &alps::hdf5::archive::is_complex)
        .def("is_null", &alps::hdf5::archive::is_null)
        .def("list_children", &alps::hdf5::archive::list_children)
        .def("list_attributes", &alps::hdf5::archive::list_attributes)
        .def("__setitem__", &alps::detail::python_hdf5_save)
        .def("__getitem__", &alps::detail::python_hdf5_load)
        .def("create_group", &alps::hdf5::archive::create_group)
        .def("delete_data", &alps::hdf5::archive::delete_data)
        .def("delete_group", &alps::hdf5::archive::delete_group)
        .def("delete_attribute", &alps::hdf5::archive::delete_attribute)
    ;
}
