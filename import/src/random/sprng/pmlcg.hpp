/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#ifndef ALPS_RANDOM_SPRNG_PMLCG_HPP
#define ALPS_RANDOM_SPRNG_PMLCG_HPP

#ifdef ALPS_DOXYGEN

namespace alps { namespace random { namespace sprng {
  /// @brief Wrapper for the SPRNG PMLCG random number generator
  ///
  ///  Wrapper for the SPRNG pmlcg random number generator

  class pmlcg;
}}}

#else

#define ALPS_SPRNG_GENERATOR   pmlcg
#define ALPS_SPRNG_TYPE        5
#define ALPS_SPRNG_MAX_STREAMS (1<<30)
#define ALPS_SPRNG_MAX_PARAMS  1

#include <alps/random/sprng/detail/implementation.hpp>

#undef ALPS_SPRNG_GENERATOR
#undef ALPS_SPRNG_TYPE
#undef ALPS_SPRNG_MAX_STREAMS
#undef ALPS_SPRNG_MAX_PARAMS

#endif

#endif // ALPS_RANDOM_SPRNG_PMLCG_HPP
