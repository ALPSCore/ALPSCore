/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
