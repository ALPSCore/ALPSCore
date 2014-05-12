/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

/// @file A convenience header includign all SPRNG libraries

#ifndef ALPS_RANDOM_SPRNG_HPP
#define ALPS_RANDOM_SPRNG_HPP

#include <alps/random/sprng/lfg.hpp>
#include <alps/random/sprng/lcg.hpp>
#include <alps/random/sprng/lcg64.hpp>
#include <alps/random/sprng/cmrg.hpp>
#include <alps/random/sprng/mlfg.hpp>
// PMLCG needs GNU multiprecission library
#ifdef ALPS_RANDOM_SPRNG_HAVEGMP
#include <boost/random/sprng/pmlcg.hpp>
#endif

#endif // ALPS_RANDOM_SPRNG_HPP
