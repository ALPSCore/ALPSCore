/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_ISNAN_HPP
#define ALPS_NUMERIC_ISNAN_HPP

#include <cmath>
#include <alps/config.hpp>

namespace alps { namespace numeric {

#ifdef isnan
#undef isnan
#endif

#if defined(BOOST_MSVC)
  template <class T>
  bool isnan(T x) { return _isnan(x);}
#elif defined(__INTEL_COMPILER) || defined(_CRAYC) || defined(__FCC_VERSION)
  using ::isnan;
#else
  using std::isnan;
#endif

} } // end namespace

#endif // ALPS_NUMERIC_ISNAN_HPP
