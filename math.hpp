/***************************************************************************
* ALPS++ library
*
* alps/math.h
*
* $Id$
*
* Copyright (C) 1999-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_MATH_HPP
#define ALPS_MATH_HPP

#include <alps/config.h>
#include <alps/typetraits.h>

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstddef>

namespace alps {

template <class T>
T abs (T x) { return std::fabs(x);}

template <class T>
T abs (std::complex<T> x) { return std::abs(x);}

inline std::size_t binomial(std::size_t l,std::size_t n)
{
  double nominator=1;
  double denominator=1;
  std::size_t n2=std::max(n,l-n);
  std::size_t n1=std::min(n,l-n);
  for (std::size_t i=n2+1;i<=l;i++)
    nominator*=i;
  for (std::size_t i=2;i<=n1;i++)
    denominator*=i;
  return std::size_t(nominator/denominator+0.1);
}

template <class T>
typename TypeTraits<T>::norm_t abs2(T x) {
  return alps::abs(x)*alps::abs(x);	
}

template <class T>
T abs2(std::complex<T> x) {
  return x.real()*x.real()+x.imag()*x.imag();
}

} // end namespace

#endif // ALPS_MATH_HPP
