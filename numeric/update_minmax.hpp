/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
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

#ifndef ALPS_NUMERIC_UPDATE_MINMAX_HPP
#define ALPS_NUMERIC_UPDATE_MINMAX_HPP

#include <alps/type_traits/slice.hpp>

namespace alps { namespace numeric {

template <class T>
void update_max(T& lhs, T const& rhs)
{
  for (typename slice_index<T>::type it = slices(lhs).first; 
       it < slices(lhs).second && it < slices(rhs).second; ++it)
    if (slice_value(lhs,it) < slice_value(rhs,it))
      slice_value(lhs,it) = slice_value(rhs,it);
}

template <class T>
void update_min(T& lhs, T const& rhs)
{
  for (typename slice_index<T>::type it = slices(lhs).first; 
       it < slices(lhs).second && it < slices(rhs).second; ++it)
    if (slice_value(rhs,it) < slice_value(lhs,it))
      slice_value(lhs,it) = slice_value(rhs,it);
}



} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_UPDATE_MINMAX_HPP
