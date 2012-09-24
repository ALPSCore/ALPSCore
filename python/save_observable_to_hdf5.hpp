/*****************************************************************************
 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations
 *
 * ALPS Libraries
 *
 * Copyright (C) 2010 by Matthias Troyer <troyer@comp-phys.org>,
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

/* $Id: make_copy.hpp 4059 2010-03-29 08:36:25Z troyer $ */

#ifndef ALPS_PYTHON_VERY_LONG_FILENAME_FOR_SAVE_OBSERVABLE_TO_HDF5_HPP
#define ALPS_PYTHON_VERY_LONG_FILENAME_FOR_SAVE_OBSERVABLE_TO_HDF5_HPP

#include <alps/hdf5.hpp>

namespace alps { namespace python {
    
    template <typename Obs> void save_observable_to_hdf5(Obs const & obs, std::string const & filename) {
        hdf5::archive ar(filename, "a");
        ar["/simulation/results/"+obs.representation()] << obs;
    }
        
} } // end namespace alps::python

#endif // ALPS_PYTHON_VERY_LONG_FILENAME_FOR_SAVE_OBSERVABLE_TO_HDF5_HPP
