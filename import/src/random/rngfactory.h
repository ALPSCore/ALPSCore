/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file rngfactory.h
/// \brief a factory to create random number generators from their name

#ifndef ALPS_RANDOMFACTORY_H
#define ALPS_RANDOMFACTORY_H

#include <alps/random/buffered_rng.h>
#include <alps/factory.h>
#include <string>

namespace alps {

/// a factory to create random number generators from their name
// \sa rng_factory
class RNGFactory : public factory<std::string,buffered_rng_base>
{
public:
  RNGFactory();
  template <class RNG> void register_rng(const std::string& name) 
  { register_type<buffered_rng<RNG> >(name);}
};


/// \brief a factory to create random number generators from their name
/// 
/// currently the folloowing two boost generators can be created from their name
/// - lagged_fibonacci607
/// - mt19937
extern RNGFactory rng_factory;

} // end namespace

#endif // ALPS_RANDOMFACTORY_H
