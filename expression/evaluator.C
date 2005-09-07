/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/expression/evaluator.h>

namespace alps {

Disorder::random_type Disorder::rng_;

int Disorder::last_seed_;
  
boost::variate_generator<Disorder::random_type&, boost::uniform_real<> > 
  Disorder::random(Disorder::rng_, boost::uniform_real<>());
    
boost::variate_generator<Disorder::random_type&, boost::normal_distribution<> > 
  Disorder::gaussian_random(Disorder::rng_, boost::normal_distribution<>());

void Disorder::seed(unsigned int i) 
{ 
  seed_with_sequence(rng_,i);
  last_seed_=i;
}

void Disorder::seed_if_unseeded(const alps::Parameters& p) 
{
  if (static_cast<int>(p.value_or_default("DISORDERSEED",0)) != last_seed_)
    seed(p.value_or_default("DISORDERSEED",0));
}

} // end namespace alps
