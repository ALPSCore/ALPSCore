/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NUMERIC_OUTER_PRODUCT_HPP
#define ALPS_NUMERIC_OUTER_PRODUCT_HPP

#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/element_type.hpp>
#include <alps/type_traits/is_sequence.hpp>
#include <alps/type_traits/covariance_type.hpp>

#include <complex>
#include <type_traits>

namespace alps { namespace numeric {

template <class T>
inline
typename std::enable_if<!is_sequence<T>::value,typename covariance_type<T>::type>::type
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
typename std::enable_if<is_sequence<T>::value,typename covariance_type<T>::type>::type
outer_product(T a, T b)
{
    throw std::logic_error("Outer product beween vectors is not implemented. "
                           "Please use the new ALEA library if you need vector-vector covariance!");
}

} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_OUTER_PRODUCT_HPP
