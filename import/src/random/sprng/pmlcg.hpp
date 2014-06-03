/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
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
