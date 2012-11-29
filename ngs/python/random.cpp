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

#define PY_ARRAY_UNIQUE_SYMBOL pyngsrandom_PyArrayHandle

#include <alps/ngs/hdf5.hpp>
#include <alps/ngs/boost_python.hpp>

#include <alps/python/make_copy.hpp>

#include <boost/random.hpp>

#include <string>

namespace alps {
    namespace detail {

        struct random01 : public boost::variate_generator<boost::mt19937, boost::uniform_01<double> > {
            random01(int seed = 0)
                : boost::variate_generator<boost::mt19937, boost::uniform_01<double> >(boost::mt19937(seed), boost::uniform_01<double>())
            {}

            void save(alps::hdf5::archive & ar) const {
                std::ostringstream os;
                os << this->engine();
                ar["engine"] << os.str();
            }

            void load(alps::hdf5::archive & ar) {
                std::string state;
                ar["engine"] >> state;
                std::istringstream is(state);
                is >> this->engine();
            }
        };

    }
}

BOOST_PYTHON_MODULE(pyngsrandom_c) {

    boost::python::class_<alps::detail::random01>(
        "random01",
        boost::python::init<boost::python::optional<int> >()
    )
        .def("__deepcopy__",  &alps::python::make_copy<alps::detail::random01>)
        .def("__call__", static_cast<alps::detail::random01::result_type(alps::detail::random01::*)()>(&alps::detail::random01::operator()))
        .def("save", &alps::detail::random01::save)
        .def("load", &alps::detail::random01::load)
    ;
}
