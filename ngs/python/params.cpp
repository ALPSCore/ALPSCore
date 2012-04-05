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

#define PY_ARRAY_UNIQUE_SYMBOL pyngsparams_PyArrayHandle

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/params.hpp>

#include <alps/python/make_copy.hpp>

#include <string>

namespace alps {
    namespace detail {

        std::size_t params_len(alps::params const & self) {
            return self.size();
        }

        boost::python::object params_getitem(alps::params & self, boost::python::object const & key) {
			return self[boost::python::call_method<std::string>(key.ptr(), "__str__")];
        }

        void params_setitem(alps::params & self, boost::python::object const & key, boost::python::object & value) {
			self[boost::python::call_method<std::string>(key.ptr(), "__str__")] = value;
        }
/*
        void params_delitem(alps::params & self, boost::python::object const & key) {
            // TODO: implement for non params_impl_dict prams
            dynamic_cast<params_impl_dict &>(*self.get_impl()).native_delitem(key);
        }
*/
        bool params_contains(alps::params & self, boost::python::object const & key) {
			return self.defined(boost::python::call_method<std::string>(key.ptr(), "__str__"));
        }

// TODO: implement!
/*        boost::python::object params_iter(alps::params & self) {
          // TODO: implement for non params_impl_dict prams
            return dynamic_cast<params_impl_dict &>(*self.get_impl()).native_iter();
        }
*/
        boost::python::object value_or_default(alps::params & self, boost::python::object const & key, boost::python::object const & value) {
			return params_contains(self, key) ? params_getitem(self, key) : value;
		}

        void params_load(alps::params & self, alps::hdf5::archive & ar, std::string const & path = "/parameters") {
            std::string current = ar.get_context();
            ar.set_context(path);
            self.load(ar);
            ar.set_context(current);
        }
        BOOST_PYTHON_FUNCTION_OVERLOADS(params_load_overloads, params_load, 2, 3)
    }
}

BOOST_PYTHON_MODULE(pyngsparams_c) {

    boost::python::class_<alps::params>(
        "params",
        boost::python::init<boost::python::object>()
    )
        .def("__len__", &alps::detail::params_len)
        .def("__deepcopy__", &alps::python::make_copy<alps::params>)
        .def("__getitem__", &alps::detail::params_getitem)
        .def("__setitem__", &alps::detail::params_setitem)
// TODO: implement
//        .def("__delitem__", &alps::detail::params_delitem)
        .def("__contains__", &alps::detail::params_contains)
// TODO: implement
//        .def("__iter__", &alps::detail::params_iter)
        .def("valueOrDefault", &alps::detail::value_or_default)
        .def("save", &alps::params::save)
        .def("load", &alps::detail::params_load, alps::detail::params_load_overloads())
    ;
}
