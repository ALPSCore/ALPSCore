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

#ifndef ALPS_NUMERIC_IS_ZERO_HPP
#define ALPS_NUMERIC_IS_ZERO_HPP

#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>

namespace alps { namespace numeric {

namespace detail {

template<class T, unsigned int N = 5>
struct precision
{
  static inline T epsilon() { return 1e-50; }
};
template<class T> struct precision<T, 0>;
template<class T> struct precision<T, 1>
{
  static inline T epsilon() { return 1e-10; }
};
template<class T> struct precision<T, 2>
{
  static inline T epsilon() { return 1e-20; }
};
template<class T> struct precision<T, 3>
{
  static inline T epsilon() { return 1e-30; }
};
template<class T> struct precision<T, 4>
{
  static inline T epsilon() { return 1e-40; }
};

} // end namespace detail


//
// is_zero
//

/// \brief checks if a number is zero
/// in case of a floating point number, absolute values less than
/// epsilon (1e-50 by default) count as zero
/// \return returns true if the value is zero
template<unsigned int N, class T>
inline bool is_zero(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return std::abs(x) < detail::precision<T, N>::epsilon(); }
template<unsigned int N, class T>
inline bool is_zero(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0)
{ return x == T(0); }

namespace detail {

template<unsigned int N, class T>
struct is_zero_helper
{
  static bool result(const T& x) { return x == T(0); }
};
template<unsigned int N, class T>
struct is_zero_helper<N, std::complex<T> >
{
  static bool result(const std::complex<T>& x)
  { return is_zero<N>(std::abs(x)); }
};

} // end namespace detail

template<unsigned int N, class T>
inline bool is_zero(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return detail::is_zero_helper<N, T>::result(x); }

template<class T>
inline bool is_zero(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return std::abs(x) < detail::precision<T>::epsilon(); }
template<class T>
inline bool is_zero(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::enable_if<boost::is_integral<T> >::type* = 0)
{ return x == T(0); }
template<class T>
inline bool is_zero(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x == T(0); }
template<class T>
inline bool is_zero(const std::complex<T>& x)
{ return is_zero(std::abs(x)); }

} } // end namespace

#endif // ALPS_NUMERIC_IS_ZERO_HPP
