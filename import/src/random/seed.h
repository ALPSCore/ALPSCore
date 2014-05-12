/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Mario Ruetti <mruetti@gmx.net>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

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
