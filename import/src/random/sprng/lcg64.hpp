/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_RANDOM_SPRNG_LCG64_HPP
#define ALPS_RANDOM_SPRNG_LCG64_HPP

/// \file alps/random/sprng/lcg_64.hpp
///
/// A wrapper for the SPRNG 64-bit linear congruential generator

#ifdef ALPS_DOXYGEN

namespace alps { namespace random { namespace sprng {
  /// @brief Wrapper for the SPRNG lcg64 random number generator
  ///
  ///  Wrapper for the SPRNG lcg64 random number generator

  class lcg64;
}}}

#else


#define ALPS_SPRNG_GENERATOR   lcg64
#define ALPS_SPRNG_TYPE        2
#define ALPS_SPRNG_MAX_STREAMS 146138719
#define ALPS_SPRNG_MAX_PARAMS  3
#define ALPS_SPRNG_VALIDATION 0.78712665431950790129

#include <alps/random/sprng/detail/implementation.hpp>

#undef ALPS_SPRNG_GENERATOR
#undef ALPS_SPRNG_TYPE
#undef ALPS_SPRNG_MAX_STREAMS
#undef ALPS_SPRNG_MAX_PARAMS
#undef ALPS_SPRNG_VALIDATION

#endif

#endif // ALPS_RANDOM_SPRNG_LCG64_HPP
