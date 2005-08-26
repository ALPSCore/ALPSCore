/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

/// \file math.hpp
/// \brief basic math functions
///
/// This header contains mathematical functions not present in the
/// standard or boost libraries.

#ifndef ALPS_MATH_HPP
#define ALPS_MATH_HPP

#include <alps/config.h>
#include <alps/typetraits.h>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>
#include <limits>

namespace alps {
namespace detail {

/// implementation detail to test whether a number is close enough to zero to truncate it, version for floating point numbers
template <bool F>
struct is_zero_float
{
  template <class T>
  static bool is_zero(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
  {
    return std::abs(x) < 1e-50; // std::sqrt(std::numeric_limits<T>::min())
  }
  template <class T>
  static bool is_zero(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
  {
    return std::abs(x) < 1e-50; // std::sqrt(std::numeric_limits<T>::min())
  }
};

/// implementation class to test whether a number is close enough to zero to truncate it, version for integers
template <>
struct is_zero_float<false>
{
  template <class T>
  static bool is_zero(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
  { return x == T(0.); }
  template <class T>
  static bool is_zero(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
  { return x == T(0.); }
};

} // end namespace detail

/// \brief calculate the binomial coefficient
/// \return the binomial coefficient l over n
inline std::size_t binomial(std::size_t l, std::size_t n)
{
  double nominator=1;
  double denominator=1;
  std::size_t n2=std::max(n,l-n);
  std::size_t n1=std::min(n,l-n);
  for (std::size_t i=n2+1;i<=l;i++)
    nominator*=i;
  for (std::size_t i=2;i<=n1;i++)
    denominator*=i;
  return std::size_t(nominator/denominator+0.1);
}

/// \brief calculate the square of the absolute value. 
/// It is optimized by specialization for complex numbers.
/// \return the square of the absolute value of the argument

template <class T>
inline typename type_traits<T>::norm_t abs2(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0) {
  return x * x;        
}
template <class T>
inline typename type_traits<T>::norm_t abs2(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0) {
  return std::abs(x)*std::abs(x);        
}

template <class T>
inline T abs2(const std::complex<T>& x) {
  return x.real()*x.real()+x.imag()*x.imag();
}

/// \brief checks if a number is zero
/// in case of a floating point number, absolute values less than 1e-50 count as zero
/// \return returns true if the value is zero
template<class T>
inline bool is_zero(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return detail::is_zero_float<boost::is_float<T>::value>::is_zero(x); }

template<class T>
inline bool is_zero(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return detail::is_zero_float<boost::is_float<T>::value>::is_zero(x); }

template<class T>
inline bool is_zero(const std::complex<T>& x) { return is_zero(std::abs(x)); }
 
/// \brief checks if a number is not zero
/// in case of a floating point number, absolute values less than 1e-50 count as zero
/// \return returns true if the value is not zero
template<class T>
inline bool is_nonzero(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0) { return !is_zero(x); }

template<class T>
inline bool is_nonzero(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0) { return !is_zero(x); }

//
// round
//

/// \brief rounds a floating point value to be exactly zero if it is nearly zero
///
/// the function is specialized for floating point and complex types and does nothing for other types
/// \return 0. if the floating point value of the argument is less than 1e-12, and the argument itself otherwise
template<class T>
inline T round(T x, typename boost::enable_if<boost::is_float<T> >::type* = 0)
{ return (std::abs(x) < 1.0e-12) ? T(0.) : x; }

/// \brief rounding of non-floating point numbers is a no-op
/// \return the unmodified argument
template<class T>
inline T round(T x, typename boost::disable_if<boost::is_float<T> >::type* = 0, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x; }

template<class T>
inline T round(const T& x, typename boost::disable_if<boost::is_float<T> >::type* = 0, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x; }

template<class T>
inline std::complex<T> round(const std::complex<T>& x)
{ return std::complex<T>(round(x.real()), round(x.imag())); }

} // end namespace

#endif // ALPS_MATH_HPP
