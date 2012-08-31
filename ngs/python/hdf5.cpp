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
#include <alps/ngs/hdf5/vector.hpp>
#include <alps/ngs/hdf5/python.hpp>
#include <alps/ngs/hdf5/complex.hpp>

#include <alps/ngs/stacktrace.hpp>
#include <alps/ngs/boost_python.hpp>
#include <alps/ngs/detail/numpy_import.ipp>

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

    boost::python::class_<alps::hdf5::archive>(
          "hdf5_archive_impl",
          boost::python::init<std::string, std::string>()
    )
        .def("__deepcopy__", &alps::python::make_copy<alps::hdf5::archive>)
        .add_property("filename", &alps::detail::python_hdf5_get_filename)
        .add_property("context", &alps::hdf5::archive::get_context)
        .def("set_context", &alps::hdf5::archive::set_context)
        .def("is_group", &alps::hdf5::archive::is_group)
        .def("is_data", &alps::hdf5::archive::is_data)
        .def("is_attribute", &alps::hdf5::archive::is_attribute)
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
