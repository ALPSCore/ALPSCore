/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#ifndef ALPS_RANDOM_SPRNG_MLFG_HPP
#define ALPS_RANDOM_SPRNG_MLFG_HPP

#ifdef ALPS_DOXYGEN

namespace alps { namespace random { namespace sprng {
  /// @brief Wrapper for the SPRNG PMLCG random number generator
  ///
  ///  Wrapper for the SPRNG mlfg random number generator

  class mlfg;
}}}

#else


#define ALPS_SPRNG_GENERATOR   mlfg
#define ALPS_SPRNG_TYPE        4
#define ALPS_SPRNG_MAX_STREAMS 0x7fffffff
#define ALPS_SPRNG_MAX_PARAMS  11
#define ALPS_SPRNG_VALIDATION 0.77628973267217871168

#include <alps/random/sprng/detail/implementation.hpp>

#undef ALPS_SPRNG_GENERATOR
#undef ALPS_SPRNG_TYPE
#undef ALPS_SPRNG_MAX_STREAMS
#undef ALPS_SPRNG_MAX_PARAMS
#undef ALPS_SPRNG_VALIDATION

#endif

#endif // ALPS_RANDOM_SPRNG_MLFG_HPP
