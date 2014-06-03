/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file seed.h
/// \brief a generic seeding function for random number generators

#ifndef ALPS_RANDOM_SEED_H
#define ALPS_RANDOM_SEED_H

#include <cmath>
#include <alps/config.h>
#include <alps/random/pseudo_des.h>

#include <boost/utility.hpp>
#include <boost/throw_exception.hpp>
#include <boost/generator_iterator.hpp>

#include <iostream>

namespace alps {

/// \brief a generic seeding function for random number generators
/// \param rng the random number generator to be seeded
/// \param seed the seed block number
///
/// seeds a random number generator following the Boost specifications
/// for such generators with a unique sequence of random numbers
/// obtained from the specified seed with the pseudo_des
/// generator. This function is useful to prepare seed blocks for
/// Monte Carlo simulations or similar applications.
template <class RNG>
void seed_with_sequence(RNG& rng, uint32_t seed)
{
  pseudo_des start(seed);
  pseudo_des end(seed);
  start(); // make start!=end
  typedef boost::generator_iterator_generator<pseudo_des>::type iterator_type;
  iterator_type start_it(boost::make_generator_iterator(start));
  iterator_type end_it(boost::make_generator_iterator(end));
  rng.seed(start_it,end_it);
}

template<class RNG, class INIGEN>
void seed_with_generator(RNG& rng, INIGEN& inigen) {
  INIGEN end(inigen);
  inigen(); // make start != end
  typedef typename boost::generator_iterator_generator<INIGEN>::type iterator_type;
  iterator_type start_it(boost::make_generator_iterator(inigen));
  iterator_type end_it(boost::make_generator_iterator(end));
  rng.seed(start_it, end_it);
}

} // end namespace

#endif // ALPS_RANDOM_SEED_H
