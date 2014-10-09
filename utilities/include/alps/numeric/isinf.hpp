/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_ISINF_HPP
#define ALPS_NUMERIC_ISINF_HPP

#include <alps/config.hpp>
#include <cmath>

namespace alps { namespace numeric {

#ifdef isinf
#undef isinf
#endif

#if defined( BOOST_MSVC)
  template <class T>
  bool isinf(T x) { return !_finite(x) && !_isnan(x);}
#elif defined(__INTEL_COMPILER) || defined(_CRAYC) || defined(__FCC_VERSION)
  using ::isinf;
#else
  using std::isinf;
#endif

} } // end namespace

#endif // ALPS_NUMERIC_ISINF_HPP
