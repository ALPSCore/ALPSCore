/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/* $Id$ */

#ifndef ALPS_ALEA_HDF5_H
#define ALPS_ALEA_HDF5_H

#include <alps/config.h>

#ifdef ALPS_HAVE_VALARRAY
#include <valarray>
#endif

#ifdef ALPS_HAVE_HDF5
# include <H5Cpp.h>
using namespace H5;

namespace alps {

template<class T>
struct HDF5Traits 
{
  static PredType pred_type() 
  {
    boost::throw_exception(runtime_error("HDF5Traits not implemented for this type"));
    return PredType::NATIVE_INT; // dummy return
  }
};

template<>
struct HDF5Traits<double> 
{
  static PredType pred_type() { return PredType::NATIVE_DOUBLE; } 
};

template<>
struct HDF5Traits<int>    
{
  static PredType pred_type() { return PredType::NATIVE_INT; } 
};

#ifdef ALPS_HAVE_VALARRAY
template<class T>
struct HDF5Traits<std::valarray<T> > 
{
  static PredType pred_type() { return HDF5Traits<T>::pred_type(); } 
};
#endif // ALPS_HAVE_VALARRAY

} // end namespace alps

#endif // ALPS_HAVE_HDF5

#endif // ALPS_ALEA_HDF5_H
