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

#ifndef ALPS_VECTORMATH_H
#define ALPS_VECTORMATH_H

#include <alps/config.h>
#include <alps/functional.h>
#include <alps/typetraits.h>
#include <vector>

namespace alps {

/// \addtogroup alps
/// @{

/// \file vectormath.h
/// \brief basic arithmetic operations on std::vectors
/// 
/// This header contains slow but simple implementations of basic arithmetic operations on std::vectors 

/// \brief apply a binary function object to two vectors
/// \param op the binary function object
/// \param x the first argument
/// \param y the second argument
/// the resulting vector is calculated by element-wise calculating op(x[i],y[i]). The two vectors should be of the same size.
template <class T, class OP>
std::vector<T> vector_vector_apply(OP op, const std::vector<T>& x, const std::vector<T>& y)
{
  typedef typename std::vector<T>::size_type size_type;
  size_type end=std::min(x.size(),y.size());
  std::vector<T> res(std::max(x.size(),y.size()));
  for (size_type i=0;i<end;++i)
    res[i]=op(x[i],y[i]);
  for (size_type i=end;i<x.size();++i)
    res[i]=op(x[i],T());
  for (size_type i=end;i<y.size();++i)
    res[i]=op(T(),y[i]);
  return res;
}

/// \brief apply a binary function object to a scalar and a vector
/// \param op the binary function object
/// \param x the scalar argument
/// \param y the vector argument
/// the resulting vector is calculated by element-wise calculating op(x,y[i])
template <class T, class S, class OP>
std::vector<T> scalar_vector_apply(OP op, S x, const std::vector<T>& y)
{
  typedef typename std::vector<T>::size_type size_type;
  std::vector<T> res(y.size());
  std::transform(y.begin(),y.end(),res.begin(),std::bind1st(op,x));
  return res;
}

/// @}
} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace std {
#endif

/// \addtogroup alps
/// @{

/// returns the sum of two vectors
template <class T>
std::vector<T> operator+(const std::vector<T>& x, const std::vector<T>& y)
{
  return alps::vector_vector_apply(std::plus<T>(),x,y);
}

/// returns the difference of two vectors
template <class T>
std::vector<T> operator-(const std::vector<T>& x, const std::vector<T>& y)
{
  return alps::vector_vector_apply(std::minus<T>(),x,y);
}

/// returns the negated vector
template <class T>
std::vector<T> operator-(const std::vector<T>& x)
{
  std::vector<T> res(x.size());
  std::transform(x.begin().x.end(),res.begin(),std::negate<T>());
  return res;
}

/// returns the vector scaled by a factor
/// \param s the scalar factor
/// \param v the vector
template <class T, class S>
std::vector<T> operator*(S s, const std::vector<T>& v)
{
  return alps::scalar_vector_apply(alps::multiplies<S,T,T>(),s,v);
}

/// returns the vector scaled by a factor
/// \param s the scalar factor
/// \param v the vector
template <class T, class S>
std::vector<T> operator*(const std::vector<T>& v, S s)
{
  return alps::scalar_vector_apply(alps::multiplies<S,T,T>(),s,v);
}

/// @}
#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace std
#endif

#endif // ALPS_VECTORMATH_H
