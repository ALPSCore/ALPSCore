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

#define ALPS_HDF5_CLOSE_GREEDY

#include <alps/hdf5.hpp>
#include <alps/python/numpy_array.hpp>

#include <boost/python.hpp>

#include <numpy/arrayobject.h>

namespace alps { 
    namespace hdf5 {

        std::string extrace_path(boost::python::object const & path) {
            boost::python::extract<std::string> path_(path);
            if (!path_.check()) {
                PyErr_SetString(PyExc_TypeError, "Invalid path");
                boost::python::throw_error_already_set();
            }
            return path_();
        }

        template<typename Archive> struct pyarchive {
            public:

                pyarchive(std::string const & filename): filename_(filename) {
                    if (mem.find(filename_) == mem.end())
                        mem[filename_] = std::make_pair(new Archive(filename_), 1);
                    else
                        ++mem[filename_].second;
                }

                virtual ~pyarchive() {
                    if (!--mem[filename_].second) {
                        delete mem[filename_].first;
                        mem.erase(filename_);
                    }
                }

                boost::python::object is_group(boost::python::object const & path) const {
                    return boost::python::object(mem[filename_].first->is_group(extrace_path(path)));
                }

                boost::python::object is_data(boost::python::object const & path) const {
                    return boost::python::object(mem[filename_].first->is_data(extrace_path(path)));
                }

                boost::python::object is_attribute(boost::python::object const & path) const {
                    return boost::python::object(mem[filename_].first->is_attribute(extrace_path(path)));
                }

                boost::python::list extent(boost::python::object const & path) const {
                    return boost::python::list(mem[filename_].first->extent(extrace_path(path)));
                }

                boost::python::object dimensions(boost::python::object const & path) const {
                    return boost::python::object(mem[filename_].first->dimensions(extrace_path(path)));
                }

                boost::python::object is_scalar(boost::python::object const & path) const {
                    return boost::python::object(mem[filename_].first->is_scalar(extrace_path(path)));
                }

                boost::python::object is_null(boost::python::object const & path) const {
                    return boost::python::object(mem[filename_].first->is_null(extrace_path(path)));
                }

                boost::python::list list_children(boost::python::object const & path) const {
                    return boost::python::list(mem[filename_].first->list_children(extrace_path(path)));
                }

                boost::python::list list_attr(boost::python::object const & path) const {
                    return boost::python::list(mem[filename_].first->list_attr(extrace_path(path)));
                }

            protected:

                std::string filename_;
                static std::map<std::string, std::pair<Archive *, std::size_t> > mem;

        };

        template<typename Archive> std::map<std::string, std::pair<Archive *, std::size_t> > pyarchive<Archive>::mem;

        struct pyoarchive : public pyarchive<alps::hdf5::oarchive> {
            public:

                pyoarchive(std::string const & filename): pyarchive<alps::hdf5::oarchive>(filename) {}

                void write(boost::python::object path, boost::python::object const & data) {
                    alps::python::import_numpy_array();
                    std::size_t size = PyArray_Size(data.ptr());
                    double * data_ = static_cast<double *>(PyArray_DATA(data.ptr()));
                    using namespace alps;
                    *mem[filename_].first << make_pvp(extrace_path(path), data_, size);
                }
        };

        struct pyiarchive : public pyarchive<alps::hdf5::iarchive> {
            public:

                pyiarchive(std::string const & filename): pyarchive<alps::hdf5::iarchive>(filename) {}

                boost::python::numeric::array read(boost::python::object const & path) {
                    alps::python::import_numpy_array();
                    std::vector<double> data;
                    *mem[filename_].first >> make_pvp(extrace_path(path), data);
                    npy_intp size = data.size();
                    boost::python::object obj(boost::python::handle<>(PyArray_SimpleNew(1, &size, PyArray_DOUBLE)));
                    void * data_ = PyArray_DATA((PyArrayObject *)obj.ptr());
                    memcpy(data_, &data.front(), PyArray_ITEMSIZE((PyArrayObject *)obj.ptr()) * size);
                    return boost::python::extract<boost::python::numeric::array>(obj);
                }

        };
    }
}

using namespace boost::python;
using namespace alps::hdf5;

BOOST_PYTHON_MODULE(pyhdf5_c) {

    class_<pyoarchive>("h5OAr", init<std::string>())
        .def("is_group", &pyoarchive::is_group)
        .def("is_data", &pyoarchive::is_data)
        .def("is_attribute", &pyoarchive::extent)
        .def("extent", &pyoarchive::dimensions)
        .def("dimensions", &pyoarchive::dimensions)
        .def("is_scalar", &pyoarchive::is_scalar)
        .def("is_null", &pyoarchive::is_null)
        .def("list_children", &pyoarchive::list_children)
        .def("write", &pyoarchive::write)
    ;

    class_<pyiarchive>("h5IAr", init<std::string>())
        .def("is_group", &pyiarchive::is_group)
        .def("is_data", &pyiarchive::is_data)
        .def("is_attribute", &pyiarchive::extent)
        .def("extent", &pyiarchive::dimensions)
        .def("dimensions", &pyiarchive::dimensions)
        .def("is_scalar", &pyiarchive::is_scalar)
        .def("is_null", &pyiarchive::is_null)
        .def("list_children", &pyiarchive::list_children)
        .def("read", &pyiarchive::read)
    ;

}
