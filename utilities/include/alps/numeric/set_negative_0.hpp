/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUEMRIC_SET_NEGATIVE_0_HPP
#define ALPS_NUEMRIC_SET_NEGATIVE_0_HPP

#include <alps/type_traits/is_sequence.hpp>
#include <complex>
#include <type_traits>

namespace alps { namespace numeric {

template <class T>
inline typename std::enable_if<!is_sequence<T>::value,void>::type
set_negative_0(T& x)
{
  if (x<T())
    x=T();
}

template <class T>
inline void set_negative_0(std::complex<T>& x)
{
  if (std::real(x)<0. || std::imag(x)<0.)
    x=0.;
}

template <class T>
inline typename std::enable_if<is_sequence<T>::value,void>::type
set_negative_0(T& a)
{
  for(std::size_t i=0; i!=a.size(); ++i)
    set_negative_0(a[i]);
}



} } // end namespace alps::numeric

#endif // ALPS_NUEMRIC_SET_NEGATIVE_0_HPP
