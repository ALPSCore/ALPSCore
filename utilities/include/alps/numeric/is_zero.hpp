/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_IS_ZERO_HPP
#define ALPS_NUMERIC_IS_ZERO_HPP

#include <cmath>
#include <type_traits>

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
  typename std::enable_if<std::is_arithmetic<T>::value >::type* = 0,
  typename std::enable_if<std::is_float<T>::value >::type* = 0)
{ return std::abs(x) < detail::precision<T, N>::epsilon(); }
template<unsigned int N, class T>
inline bool is_zero(T x,
  typename std::enable_if<std::is_arithmetic<T>::value >::type* = 0,
  typename std::enable_if<std::is_integral<T>::value >::type* = 0)
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
  typename std::enable_if<!std::is_arithmetic<T>::value >::type* = 0)
{ return detail::is_zero_helper<N, T>::result(x); }

template<class T>
inline bool is_zero(T x,
  typename std::enable_if<std::is_arithmetic<T>::value >::type* = 0,
  typename std::enable_if<std::is_float<T>::value >::type* = 0)
{ return std::abs(x) < detail::precision<T>::epsilon(); }
template<class T>
inline bool is_zero(T x,
  typename std::enable_if<std::is_arithmetic<T>::value >::type* = 0,
  typename std::enable_if<std::is_integral<T>::value >::type* = 0)
{ return x == T(0); }
template<class T>
inline bool is_zero(const T& x,
  typename std::enable_if<!std::is_arithmetic<T>::value >::type* = 0)
{ return x == T(0); }
template<class T>
inline bool is_zero(const std::complex<T>& x)
{ return is_zero(std::abs(x)); }

} } // end namespace

#endif // ALPS_NUMERIC_IS_ZERO_HPP
