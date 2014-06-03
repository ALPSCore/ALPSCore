/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: real.hpp 3958 2010-03-05 09:24:06Z troyer $ */

#ifndef ALPS_NUMERIC_IMAG_HPP
#define ALPS_NUMERIC_IMAG_HPP

#include <algorithm>
#include <complex>
#include <vector>

namespace alps { namespace numeric {


template <class T>
inline T imag(T x) { return T(0);}

template <class T>
inline T imag(std::complex<T> x) { return std::imag(x); }

} }  // end namespace alps::numeric

#endif // ALPS_MATH_HPP
