/***************************************************************************
* ALPS++ library
*
* alps/functional.h
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
inline T operator()(const T& a, const T& b) const {
  return a*b;
}
};

template <class T>
struct conj_mult<std::complex<T> > : 
  std::binary_function<std::complex<T>,std::complex<T>,std::complex<T> > {
inline std::complex<T> operator()(const std::complex<T>& a, const std::complex<T>& b) const {
  return std::conj(a)*b;
}
};

template<class T> 
struct add_abs : public std::binary_function<T,typename TypeTraits<T>::norm_t,typename TypeTraits<T>::norm_t> {
inline typename TypeTraits<T>::norm_t operator()(typename TypeTraits<T>::norm_t sum, T val) const { return sum+alps::abs(val);}
};

template<class T> 
struct add_abs2 : public std::binary_function<T,typename TypeTraits<T>::norm_t,typename TypeTraits<T>::norm_t> {
inline typename TypeTraits<T>::norm_t operator()(typename TypeTraits<T>::norm_t sum, T val) const { return sum+abs2(val);}
};

}

#endif // ALPS_FUNCTIONAL_H
