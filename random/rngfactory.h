/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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
