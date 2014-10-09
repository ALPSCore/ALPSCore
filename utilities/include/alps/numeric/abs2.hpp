/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_ABS2_HPP
#define ALPS_NUMERIC_ABS2_HPP

#include <complex>
#include <cmath>

namespace alps { namespace numeric {

/// \brief calculate the square of the absolute value.
/// It is optimized by specialization for complex numbers.
/// \return the square of the absolute value of the argument
template <class T>
inline typename norm_type<T>::type abs2(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0) {
  return x * x;
}
template <class T>
inline typename norm_type<T>::type abs2(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0) {
  return std::abs(x)*std::abs(x);
}

template <class T>
inline T abs2(const std::complex<T>& x) {
  return x.real()*x.real()+x.imag()*x.imag();
}


} } // end namespace

#endif // ALPS_NUMERIC_ABS2_HPP
