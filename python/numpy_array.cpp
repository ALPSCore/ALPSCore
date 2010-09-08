/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Lukas Gamper <gamperl@gmail.com>,
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

/* $Id: pyalea.cpp 3520 2010-04-09 16:49:53Z tamama $ */

#include <alps/python/numpy_array.hpp>

#include <numpy/arrayobject.h>

namespace alps { 
    namespace python {
        void import_numpy_array() {
            static bool inited = false;
            if (!inited) {
                import_array();  
                boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
                inited = true;
            }
        }
        void convert_numpy_array(boost::python::object const & source, std::vector<double> & target) {
            import_numpy_array();
            target.resize(PyArray_Size(source.ptr()));
            memcpy(&target.front(), static_cast<double *>(PyArray_DATA(source.ptr())), PyArray_ITEMSIZE((PyArrayObject *)source.ptr()) * target.size());
        }
    }
}
