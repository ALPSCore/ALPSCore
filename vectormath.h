/***************************************************************************
* ALPS++ library
*
* alps/vectormath.h   A class to store parameters
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

#ifndef ALPS_VECTORMATH_H
#define ALPS_VECTORMATH_H

#include <alps/config.h>
#include <alps/functional.h>
#include <alps/typetraits.h>
#include <vector>

namespace alps {

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

template <class T, class S, class OP>
std::vector<T> scalar_vector_apply(OP op, S x, const std::vector<T>& y)
{
  typedef typename std::vector<T>::size_type size_type;
  std::vector<T> res(y.size());
  std::transform(y.begin(),y.end(),res.begin(),std::bind1st(op,x));
  return res;
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace std {
#endif

template <class T>
std::vector<T> operator+(const std::vector<T>& x, const std::vector<T>& y)
{
  return alps::vector_vector_apply(std::plus<T>(),x,y);
}

template <class T>
std::vector<T> operator-(const std::vector<T>& x, const std::vector<T>& y)
{
  return alps::vector_vector_apply(std::minus<T>(),x,y);
}

template <class T>
std::vector<T> operator-(const std::vector<T>& x)
{
  std::vector<T> res(x.size());
  std::transform(x.begin().x.end(),res.begin(),std::negate<T>());
  return res;
}

template <class T, class S>
std::vector<T> operator*(S x, const std::vector<T>& y)
{
  return alps::scalar_vector_apply(alps::multiplies<S,T,T>(),x,y);
}

template <class T, class S>
std::vector<T> operator*(const std::vector<T>& y, S x)
{
  return alps::scalar_vector_apply(alps::multiplies<S,T,T>(),x,y);
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace std
#endif

#endif // ALPS_VECTORMATH_H
