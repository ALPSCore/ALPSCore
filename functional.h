/***************************************************************************
* ALPS++ library
*
* alps/functional.h
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#ifndef ALPS_FUNCTIONAL_H
#define ALPS_FUNCTIONAL_H

#include <alps/config.h>
#include <alps/math.hpp>
#include <complex>
#include <functional>

namespace alps {

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct plus : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x + y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct minus : std::binary_function<Arg1, Arg2, Result>  {
  Result operator () (const Arg1& x, const Arg2& y) const { return x - y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct multiplies : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x * y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct divides : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x / y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct modulus : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x % y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct bit_and : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x & y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct bit_or : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x | y; }
};

template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct bit_xor : std::binary_function<Arg1, Arg2, Result> {
  Result operator () (const Arg1& x, const Arg2& y) const { return x ^ y; }
};

template <class T, class X>
struct plus_scaled : std::binary_function<T, T, T> {
    plus_scaled(X x) : val(x) {}
    T operator () (const T& x, const T& y) const { return x + val*y; }
private:
    X val;
};

template <class T, class X>
struct minus_scaled : public std::binary_function<T, T, T> {
    minus_scaled(X x) : val(x) {}
    T operator () (const T& x, const T& y) const { return x - val*y; }
private:
    X val;
};

template <class T>
struct absmax : public std::binary_function<T, T, typename TypeTraits<T>::norm_t> {
    typename TypeTraits<T>::norm_t operator () (const T& x, const T& y) const { return std::max(alps::abs(x),alps::abs(y)); }
};

template <class T>
struct conj_mult : std::binary_function<T,T,T> {
inline T operator()(const T& a, const T& b) {
  return a*b;
}
};

template <class T>
struct conj_mult<std::complex<T> > : 
  std::binary_function<std::complex<T>,std::complex<T>,std::complex<T> > {
inline std::complex<T> operator()(const std::complex<T>& a, const std::complex<T>& b) {
  return std::conj(a)*b;
}
};

template<class T> 
struct add_abs : public std::binary_function<T,typename TypeTraits<T>::norm_t,typename TypeTraits<T>::norm_t> {
inline typename TypeTraits<T>::norm_t operator()(typename TypeTraits<T>::norm_t sum, T val) { return sum+alps::abs(val);}
};

template<class T> 
struct add_abs2 : public std::binary_function<T,typename TypeTraits<T>::norm_t,typename TypeTraits<T>::norm_t> {
inline typename TypeTraits<T>::norm_t operator()(typename TypeTraits<T>::norm_t sum, T val) { return sum+abs2(val);}
};

}

#endif // ALPS_FUNCTIONAL_H
