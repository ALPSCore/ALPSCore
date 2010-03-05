/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_NUMERIC_IS_POSITIVE_HPP
#define ALPS_NUMERIC_IS_POSITIVE_HPP

#include <alps/numeric/is_zero.hpp>

namespace alps { namespace numeric {

//
// is_positive
//

template<unsigned int N, class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return is_nonzero<N>(x) && x > T(0); }
template<unsigned int N, class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0)
{ return x > T(0); }
template<unsigned int N, class T>
inline bool is_positive(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero<N>(x) && x > T(0); }

template<class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return is_nonzero(x) && x > T(0); }
template<class T>
inline bool is_positive(T x,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0)
{ return x > T(0); }
template<class T>
inline bool is_positive(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_nonzero(x) && x > T(0); }


} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_IS_POSITIVE_HPP
