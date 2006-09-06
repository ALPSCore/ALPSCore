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

/// \file vectormath.h
/// \brief basic arithmetic operations on std::vectors
/// 
/// This header contains slow but simple implementations of basic
/// arithmetic operations on std::vectors

#ifndef ALPS_VECTORMATH_H
#define ALPS_VECTORMATH_H


#include <alps/config.h>
#include <alps/typetraits.h>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <cassert>
#include <vector>

namespace alps {

namespace detail {

/// \brief apply a binary function object to two vectors
/// \param op the binary function object
/// \param x the first argument
/// \param y the second argument
/// the resulting vector is calculated by element-wise calculating op(x[i],y[i]). The two vectors should be of the same size.
template <class T, class OP>
std::vector<T> vector_vector_apply(OP op, const std::vector<T>& x, const std::vector<T>& y)
{
  assert(x.size()==y.size());
  std::vector<T> res;
  std::transform(x.begin(),x.end(),y.begin(),std::back_inserter(res),op);
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
  using namespace boost::lambda;
  std::vector<T> res(y.size());
  for (int i=0;i<y.size();++i)
    res[i] = op(x,y[i]);
  //std::transform(y.begin(),y.end(),std::back_inserter(res),bind(op,x,_1));
  return res;
}

} // namespace detail
} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace std{
#endif

/// returns the sum of two vectors
template <class T>
std::vector<T> operator+(const std::vector<T>& x, const std::vector<T>& y)
{
  using namespace boost::lambda;
  return alps::detail::vector_vector_apply(_1 + _2,x,y);
}

/// returns the difference of two vectors
template <class T>
std::vector<T> operator-(const std::vector<T>& x, const std::vector<T>& y)
{
  using namespace boost::lambda;
  return alps::detail::vector_vector_apply(_1 - _2,x,y);
}

/// returns the negated vector
template <class T>
std::vector<T> operator-(const std::vector<T>& x)
{
  using namespace boost::lambda;
  std::vector<T> res(x.size());
  std::transform(x.begin().x.end(),res.begin(),-_1);
  return res;
}

/// returns the vector scaled by a factor
/// \param s the scalar factor
/// \param v the vector
template <class T, class S>
std::vector<T> operator*(S s, const std::vector<T>& v)
{
  using namespace boost::lambda;
  return alps::detail::scalar_vector_apply(_1 * _2,s,v);
}

/// returns the vector scaled by a factor
/// \param s the scalar factor
/// \param v the vector
template <class T, class S>
std::vector<T> operator*(const std::vector<T>& v, S s)
{
  using namespace boost::lambda;
  return alps::detail::scalar_vector_apply(_1 * _2,s,v);
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
}
#endif

#endif // ALPS_VECTORMATH_H
