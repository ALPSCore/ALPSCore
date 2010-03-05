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

#ifndef ALPS_NUMERIC_IS_EQUAL_HPP
#define ALPS_NUMERIC_IS_EQUAL_HPP

#include <alps/config.h>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <algorithm>
#include <complex>
#include <vector>
#include <cmath>
#include <cstddef>

namespace alps { namespace numeric {

//
// is_equal
//

template<unsigned int N, class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0,
  typename boost::enable_if<boost::is_integral<U> >::type* = 0)
{ return x == y; }
template<unsigned int N, class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_float<T> >::type* = 0,
  typename boost::enable_if<boost::is_float<U> >::type* = 0)
{ return is_zero<N>(x) ? is_zero<N>(y) : is_zero<N>((x-y)/x); }
template<unsigned int N, class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_float<T> >::type* = 0,
  typename boost::enable_if<boost::is_integral<U> >::type* = 0)
{ return is_equal<N>(x, T(y)); }
template<unsigned int N, class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0,
  typename boost::enable_if<boost::is_float<U> >::type* = 0)
{ return is_equal<N>(y, x); }

namespace detail {

template<unsigned int N, class T, class U>
struct is_equal_helper
{
  static bool result(const T& x, const U& y,
    typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
  { return x == y; }
  static bool result(const T& x, U y,
    typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0)
  { return x == y; }
};
template<unsigned int N, class T, class U>
struct is_equal_helper<N, std::complex<T>, std::complex<U> >
{
  static bool result(const std::complex<T>& x, const std::complex<U>& y)
  { return is_equal<N>(x.real(), y.real()) && is_equal<N>(x.imag(), y.imag()); }
};
template<unsigned int N, class T, class U>
struct is_equal_helper<N, std::complex<T>, U>
{
  static bool result(const std::complex<T>& x, const U& y,
    typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
  { return x == y; }
  static bool result(const std::complex<T>& x, U y,
    typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0)
  { return is_equal<N>(x.real(), y) && is_zero<N>(x.imag()); }
};

} // end namespace detail

template<unsigned int N, class T, class U>
inline bool is_equal(const T& x, const U& y,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
{ return detail::is_equal_helper<N, T, U>::result(x, y); }
template<unsigned int N, class T, class U>
inline bool is_equal(const T& x, U y,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0)
{ return detail::is_equal_helper<N, T, U>::result(x, y); }
template<unsigned int N, class T, class U>
inline bool is_equal(T x, const U& y,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
{ return detail::is_equal_helper<N, U, T>::result(y, x); }

template<class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0,
  typename boost::enable_if<boost::is_integral<U> >::type* = 0)
{ return x == y; }
template<class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_float<T> >::type* = 0,
  typename boost::enable_if<boost::is_float<U> >::type* = 0)
{ return is_zero(x) ? is_zero(y) : is_zero((x-y)/x); }
template<class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_float<T> >::type* = 0,
  typename boost::enable_if<boost::is_integral<U> >::type* = 0)
{ return is_equal(x, T(y)); }
template<class T, class U>
inline bool is_equal(T x, U y,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0,
  typename boost::enable_if<boost::is_float<U> >::type* = 0)
{ return is_equal(U(x), y); }
template<class T, class U>
inline bool is_equal(const T& x, const U& y,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
{ return x == y; }
template<class T, class U>
inline bool is_equal(const T& x, U y,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0)
{ return x == y; }
template<class T, class U>
inline bool is_equal(T x, const U& y,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
{ return x == y; }
template<class T, class U>
inline bool is_equal(const std::complex<T>& x, const std::complex<U>& y)
{ return is_equal(x.real(), y.real()) && is_equal(x.imag(), y.imag()); }
template<class T, class U>
inline bool is_equal(const std::complex<T>& x, U y,
  typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0)
{ return is_equal(x.real(), y) && is_zero(x.imag()); }
template<class T, class U>
inline bool is_equal(T x, const std::complex<U>& y,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_equal(y, x); }

} } // end namespace alps::numeric

#endif // ALPS_MATH_HPP
