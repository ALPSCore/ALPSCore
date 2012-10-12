/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2011 by Lukas Gamper <gamperl@gmail.com>                   *
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

#ifndef ALPS_NGS_DETAIL_EXPORT_SIM_TO_PYTHON_HPP
#define ALPS_NGS_DETAIL_EXPORT_SIM_TO_PYTHON_HPP

#include <alps/ngs/boost_python.hpp>

#include <alps/python/make_copy.hpp>

#include <boost/bind.hpp>
#include <boost/python/dict.hpp>
#include <boost/python/wrapper.hpp>
#include <boost/python/return_internal_reference.hpp>

#define ALPS_EXPORT_SIM_TO_PYTHON(NAME, CLASS)                                                                                                      \
    boost::python::class_< CLASS , boost::noncopyable, boost::python::bases<alps::mcbase_ng> >(                                                     \
          #NAME ,                                                                                                                                   \
          boost::python::init< CLASS ::parameters_type const &, boost::python::optional<std::size_t> >()                                            \
    )                                                                                                                                               \
        .add_property("params", boost::python::make_function(                                                                                       \
            static_cast<alps::mcbase_ng::parameters_type &( CLASS ::*)()>(& CLASS ::get_params), boost::python::return_internal_reference<>()       \
         ))                                                                                                                                         \
        .add_property("measurements", boost::python::make_function(                                                                                 \
            static_cast<alps::mcobservables &( CLASS ::*)()>(& CLASS ::get_measurements), boost::python::return_internal_reference<>()              \
         ))                                                                                                                                         \
        .def("run", static_cast<bool( CLASS ::*)(boost::python::object)>(& CLASS ::run))                                                            \
        .def("random", & CLASS ::random)                                                                                                            \
        .def("save", static_cast<void( CLASS ::*)(alps::hdf5::archive &) const>(& CLASS ::save))                                                    \
        .def("load", static_cast<void( CLASS ::*)(alps::hdf5::archive &)>(& CLASS ::load))                                                          \

#endif
