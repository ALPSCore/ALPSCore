/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_LATTICE_DIMENSIONAL_TRAITS_H
#define ALPS_LATTICE_DIMENSIONAL_TRAITS_H

#include <alps/config.h>
#include <boost/limits.hpp>
#include <boost/config.hpp>

namespace alps {

template <class Dimensional>
struct dimensional_traits {
  typedef std::size_t dimension_type;
  BOOST_STATIC_CONSTANT(bool, fixed_dimension=false);
  static dimension_type infinity()
  {
    return std::numeric_limits<dimension_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }
};

/*
template <class Dimensional>
inline typename dimensional_traits<Dimensional>::dimension_type
dimension(const Dimensional& d)
{
  return d.size();
}
*/


template <class T, class A>
inline typename dimensional_traits<std::vector<T,A> >::dimension_type
dimension(const std::vector<T,A>& d)
{
  return d.size();
}

/*
template <class T, int sz>
struct dimensional_traits<T[sz]> {
  typedefint dimension_type;
  BOOST_STATIC_CONSTANT(bool, fixed_dimension=true);
  BOOST_STATIC_CONSTANT(dimension_type, dimension=sz);
  static dimension_type infinity() { return=std::numeric_limits<dimension_type>::max();}
};
  
template <class T, int sz>
typename dimensional_traits<T[sz]>::dimension_type
inline dimension(const T[sz]& d)
{
  return sz;
}
*/

} // end namespace alps

#endif // ALPS_LATTICE_DIMENSIONAL_TRAITS_H
