/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#ifndef ALPS_RANDOM_SPRNG_CMRG_HPP
#define ALPS_RANDOM_SPRNG_CMRG_HPP


#ifdef ALPS_DOXYGEN

namespace alps { namespace random { namespace sprng {
  /// @brief Wrapper for the SPRNG PMLCG random number generator
  ///
  ///  Wrapper for the SPRNG cmrg random number generator

  class cmrg;
}}}

#else


#define ALPS_SPRNG_GENERATOR   cmrg
#define ALPS_SPRNG_TYPE        3
#define ALPS_SPRNG_MAX_STREAMS 146138719
#define ALPS_SPRNG_MAX_PARAMS  3
#define ALPS_SPRNG_VALIDATION 0.23551060704332388296


#include <alps/random/sprng/detail/implementation.hpp>

#undef ALPS_SPRNG_GENERATOR
#undef ALPS_SPRNG_TYPE
#undef ALPS_SPRNG_MAX_STREAMS
#undef ALPS_SPRNG_MAX_PARAMS
#undef ALPS_SPRNG_VALIDATION

#endif

#endif // ALPS_RANDOM_SPRNG_CMRG_HPP
