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

/// \file vectortraits.h
/// \brief traits classes and generic programming support for 1D container
/// 
/// This header contains traits and generic algorithms for containers.
/// It should be replaced by an extended version of boost collection traits or something similar at some time.

#ifndef ALPS_VECTORTRAITS_H
#define ALPS_VECTORTRAITS_H

#ifdef HAVE_CONFIG_H
# include <alps/config.h>
#endif
#include <alps/typetraits.h>
#include <alps/functional.h>

#include <boost/lambda/lambda.hpp>

#include <algorithm>
#include <cassert>
#include <iostream>
#include <numeric>
#include <vector>
#include <valarray>

namespace alps {

using namespace boost::lambda;

/// \brief traits for containers
/// @param CONTAINER the type of container
template <class CONTAINER> 
struct vector_traits
{
  typedef typename CONTAINER::value_type value_type;
  typedef typename CONTAINER::iterator iterator;
  typedef typename CONTAINER::const_iterator const_iterator;
  typedef typename CONTAINER::size_type size_type;
};


/// specialization of vector_traits to std::valarray<T>
template <class T> 
struct vector_traits<std::valarray<T> > {
  typedef T value_type;
  typedef T* iterator; 
  typedef const T* const_iterator; 
  typedef std::size_t size_type;
};


/// \brief the namespace for vector operations.
///  
/// This will hopefully soon be replaced by a nice Boost library
namespace vectorops {

/// returns the size of a vector
template <class C>
inline typename vector_traits<C>::size_type size(const C& c) { return c.size();}

/// returns a pointer to the start of storage of a vector
template <class C>
inline typename vector_traits<C>::value_type* data(C& c) { return &c[0];}

/// returns a pointer to the start of storage of a vector
template <class C>
inline const typename vector_traits<C>::value_type* data(const C& c) { return &c[0];}

/// resizes the vector, not necessarily keeping the contents
template <class C>
inline void resize(C& c, std::size_t n) 
{
  if(c.size()!=n)
           c.resize(n);
}

/// calculates the scalar product of two vectors
template <class C>
inline typename vector_traits<C>::value_type scalar_product(const C& c1, const C& c2) 
{
  return std::inner_product(c1.begin(),c1.end(),c2.begin(),typename C::value_type(),
                              _1+_2 ,conj_mult<typename C::value_type>());
}


/// \overload
template <class T>
inline T scalar_product(const std::valarray<T>& c1, const std::valarray<T>& c2) 
{
  return std::inner_product(data(c1),data(c1)+c1.size(),data(c2),T(), _1+_2 ,conj_mult<T>());
}


} // namespace vectorops
} // namespace alps

#endif // ALPS_VECTORTRAITS_H
