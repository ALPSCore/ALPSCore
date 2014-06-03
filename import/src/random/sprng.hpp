/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
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
