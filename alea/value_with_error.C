/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Bela Bauer <bauerb@itp.phys.ethz.ch>
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

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */


#include <boost/python.hpp>
#include "value_with_error_module.hpp"

using namespace boost::python;


BOOST_PYTHON_MODULE(value_with_error)
{
  class_<value_with_error<double> >("value_with_error",init<optional<value_with_error<double>::value_type,value_with_error<double>::value_type> >())
    .add_property("mean", &value_with_error<double>::mean)
    .add_property("error",&value_with_error<double>::error)  

    .def("__str__",  &value_with_error<double>::print_as_str)
    .def("__repr__", &value_with_error<double>::print_as_str)

    .def(+self)
    .def(-self)
    .def("__abs__", &value_with_error<double>::abs)

    .def(self += value_with_error<double>())
    .def(self += value_with_error<double>::value_type())
    .def(self -= value_with_error<double>())
    .def(self -= value_with_error<double>::value_type())
    .def(self *= value_with_error<double>())
    .def(self *= value_with_error<double>::value_type())
    .def(self /= value_with_error<double>())
    .def(self /= value_with_error<double>::value_type())

    .def(self + value_with_error<double>())
    .def(self + value_with_error<double>::value_type())
    .def(value_with_error<double>::value_type() + self)
    .def(self - value_with_error<double>())
    .def(self - value_with_error<double>::value_type())
    .def(value_with_error<double>::value_type() - self)
    .def(self * value_with_error<double>())
    .def(self * value_with_error<double>::value_type())
    .def(value_with_error<double>::value_type() * self)
    .def(self / value_with_error<double>())
    .def(self / value_with_error<double>::value_type())
    .def(value_with_error<double>::value_type() / self)

    .def("__pow__",&value_with_error<double>::pow)
    .def("sq",&value_with_error<double>::sq)
    .def("cb",&value_with_error<double>::cb)
    .def("sqrt",&value_with_error<double>::sqrt)
    .def("cbrt",&value_with_error<double>::cbrt)
    .def("exp",&value_with_error<double>::exp)
    .def("log",&value_with_error<double>::log)

    .def("sin",&value_with_error<double>::sin)
    .def("cos",&value_with_error<double>::cos)
    .def("tan",&value_with_error<double>::tan)
    .def("asin",&value_with_error<double>::asin)
    .def("acos",&value_with_error<double>::acos)
    .def("atan",&value_with_error<double>::atan)
    .def("sinh",&value_with_error<double>::sinh)
    .def("cosh",&value_with_error<double>::cosh)
    .def("tanh",&value_with_error<double>::tanh)
    .def("asinh",&value_with_error<double>::asinh)
    .def("acosh",&value_with_error<double>::acosh)
    .def("atanh",&value_with_error<double>::atanh)
    ;
}

