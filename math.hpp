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
template<unsigned int N, class T>
inline bool is_zero(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x == T(0); }
template<unsigned int N, class T>
inline bool is_zero(const std::complex<T>& x)
{ return is_zero<N>(std::abs(x)); }

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


//
// is_nonzero
//

/// \brief checks if a number is not zero
/// in case of a floating point number, absolute values less than
/// epsilon (1e-50 by default) count as zero
/// \return returns true if the value is not zero
template<unsigned int N, class T>
inline bool is_nonzero(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return !is_zero<N>(x); }
template<unsigned int N, class T>
inline bool is_nonzero(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return !is_zero<N>(x); }

template<class T>
inline bool is_nonzero(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return !is_zero(x); }
template<class T>
inline bool is_nonzero(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return !is_zero(x); }


//
// is_equal
//

template<unsigned int N, class T, class U>
inline bool is_equal(const T& x, const U& y,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
{ return x == y; }
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
template<unsigned int N, class T, class U>
inline bool is_equal(const std::complex<T>& x, const std::complex<U>& y)
{ return is_equal<N>(x.real(), y.real()) && is_equal<N>(x.imag(), y.imag()); }
template<unsigned int N, class T, class U>
inline bool is_equal(const std::complex<T>& x, U y,
  typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0)
{ return is_equal<N>(x.real(), y) && is_zero<N>(x.imag()); }
template<unsigned int N, class T, class U>
inline bool is_equal(T x, const std::complex<U>& y,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_equal<N>(y, x); }

template<class T, class U>
inline bool is_equal(const T& x, const U& y,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0,
  typename boost::disable_if<boost::is_arithmetic<U> >::type* = 0)
{ return x == y; }
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


//
// round
//

/// \brief rounds a floating point value to be exactly zero if it is nearly zero
///
/// the function is specialized for floating point and complex types
/// and does nothing for other types
/// \return 0. if the floating point value of the argument is less
/// than epsilon (1e-50 by default), and the argument itself otherwise
template<unsigned int N, class T>
inline T round(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_zero<N>(x) ? T(0) : x; }
template<unsigned int, class T>
inline T round(const T& x, 
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x; }
template<unsigned int N, class T>
inline std::complex<T> round(const std::complex<T>& x)
{ return std::complex<T>(round<N>(x.real()), round<N>(x.imag())); }

template<class T>
inline T round(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return is_zero(x) ? T(0) : x; }
template<class T>
inline T round(const T& x, 
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x; }
template<class T>
inline std::complex<T> round(const std::complex<T>& x)
{ return std::complex<T>(round(x.real()), round(x.imag())); }


//
// is_negative
//

template<class T>
inline bool is_negative(T x,
  typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x < 0; }
template<class T>
inline bool is_negative(const T& x,
  typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{ return x < 0; }
inline bool is_negative(unsigned char) { return false; }
inline bool is_negative(unsigned short) { return false; }
inline bool is_negative(unsigned int) { return false; }
inline bool is_negative(unsigned long) { return false; }

} // end namespace

#endif // ALPS_MATH_HPP
