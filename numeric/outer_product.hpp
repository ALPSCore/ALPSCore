/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
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

#ifndef ALPS_NUMERIC_OUTER_PRODUCT_HPP
#define ALPS_NUMERIC_OUTER_PRODUCT_HPP

#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/element_type.hpp>
#include <alps/type_traits/is_sequence.hpp>
#include <alps/type_traits/covariance_type.hpp>

#include <boost/utility/enable_if.hpp>

#include <complex>

namespace alps { namespace numeric {

template <class T>
inline 
typename boost::disable_if<is_sequence<T>,typename covariance_type<T>::type>::type 
outer_product(T a, T b) 
{
  return a*b;
}


template <class T>
inline std::complex<T> outer_product(std::complex<T> const& a, std::complex<T> const& b) 
{
  return std::conj(a)*b;
}


template <class T>
inline 
typename boost::enable_if<is_sequence<T>,typename covariance_type<T>::type>::type 
outer_product(T a, T b) 
{
  typedef typename average_type<typename element_type<T>::type>::type value_type;
  boost::numeric::ublas::vector<value_type> vec1(a.size()), vec2(b.size());
  for (int i=0; i<a.size(); ++i)
    vec1[i] = a[i];
  for (int i=0; i<b.size(); ++i)
    vec2[i] = b[i];
  return boost::numeric::ublas::outer_prod(vec1, vec2);
}

} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_OUTER_PRODUCT_HPP
