/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_FUNCTIONAL_H
#define ALPS_FUNCTIONAL_H

#include <alps/config.h>
#include <alps/math.hpp>
#include <complex>
#include <functional>

namespace alps {

/// \addtogroup alps
/// @{

/// \file functional.h
/// \brief extensions to the standard functional header
///
/// This header contains mathematical function objects not present in the standard
/// or boost libraries.

/// \brief extension of std::plus
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct plus : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the sum \a x + \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x + y; }
};

/// \brief extension of std::minus
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct minus : std::binary_function<Arg1, Arg2, Result>  {
/// \brief returns the difference \a x - \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x - y; }
};

/// \brief extension of std::multiplies
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct multiplies : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the product \a x * \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x * y; }
};

/// \brief extension of std::divides
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct divides : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the ratio \a x / \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x / y; }
};

/// \brief extension of std::modulus
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct modulus : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the remainder \a x % \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x % y; }
};

/// \brief extension of std::bit_and
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct bit_and : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the bitwise and \a x & \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x & y; }
};

/// \brief extension of std::bit_or
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct bit_or : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the bitwise or \a | & \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x | y; }
};

/// \brief extension of std::bit_xor
///
/// allows different types for the two arguments and the result value
/// \param Arg1 the type of the first argument
/// \param Arg2 the type of the second argument
/// \param Result the type of the result
template <class Arg1, class Arg2=Arg1, class Result=Arg1>
struct bit_xor : std::binary_function<Arg1, Arg2, Result> {
/// \brief returns the bitwise exclusive or \a ^ & \a y of the two arguments
  Result operator () (const Arg1& x, const Arg2& y) const { return x ^ y; }
};

/// \brief a function object for x+a*y
///
/// \param T the type of the arguments and result
/// \param X the type of the scale factor
template <class T, class X>
struct plus_scaled : std::binary_function<T, T, T> {
/// the scale factor is set in the constructor
/// \param a the value of the scale factor
    plus_scaled(X a) : val(a) {}
/// brief returns x+a*y
/// \return the result of  \a x + \a * \a y, where the scale factor \a a is set in the constructor
    T operator () (const T& x, const T& y) const { return x + val*y; }
private:
    X val;
};

/// \brief a function object for x-a*y
///
/// \param T the type of the arguments and result
/// \param X the type of the scale factor
template <class T, class X>
struct minus_scaled : public std::binary_function<T, T, T> {
/// the scale factor is set in the constructor
/// \param a the value of the scale factor
    minus_scaled(X a) : val(a) {}
/// brief returns x-a*y
/// \return the result of  \a x - \a * \a y, where the scale factor \a a is set in the constructor
    T operator () (const T& x, const T& y) const { return x - val*y; }
private:
    X val;
};

/// \brief a function object for max(|x|,|y|)
///
/// \param T the type of the arguments and result
template <class T>
struct absmax : public std::binary_function<T, T, typename type_traits<T>::norm_t> {
/// brief returns the maximum of the abosulte values of the two arguments
    typename type_traits<T>::norm_t operator () (const T& x, const T& y) const { return std::max(std::abs(x),std::abs(y)); }
};

/// \brief a function object for conj(x)*y
///
/// the version for real data types is just the same as std::multiplies
/// \param T the type of the arguments and result
template <class T>
struct conj_mult : std::binary_function<T,T,T> {
/// \brief returns x*y
  T operator()(const T& a, const T& b) const { return a*b; }
};

/// \brief a function object for conj(x)*y
///
/// the version for complex data types is specialized
/// \param T the type of the arguments and result
template <class T>
struct conj_mult<std::complex<T> > : 
  std::binary_function<std::complex<T>,std::complex<T>,std::complex<T> > {
/// \brief returns std::conj(x)*y
  std::complex<T> operator()(const std::complex<T>& a, const std::complex<T>& b) const {
  return std::conj(a)*b;
}
};

/// \brief a function object for x + |y|
///
/// @param T the type of the argument @a y
template<class T> 
struct add_abs : public std::binary_function<T,typename type_traits<T>::norm_t,typename type_traits<T>::norm_t> {
/// \brief returns x+|y|
/// \return the value of x+std::abs(y)
typename type_traits<T>::norm_t operator()(typename type_traits<T>::norm_t x, T y) const { return x+std::abs(y);}
};

/// \brief a function object for x + |y|^2
///
/// \param T the type of the argument \a y
template<class T> 
struct add_abs2 : public std::binary_function<T,typename type_traits<T>::norm_t,typename type_traits<T>::norm_t> {
/// \brief returns x+|y|^2
typename type_traits<T>::norm_t operator()(typename type_traits<T>::norm_t x, T y) const { return x+abs2(y);}
};

/// @}

}

#endif // ALPS_FUNCTIONAL_H
