/***************************************************************************
* ALPS++ library
*
* alps/random/rngfactory.h a factory for RNG generators from parameters or XML
*
* $Id$
*
* Copyright (C) 2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
**************************************************************************/

#ifndef ALPS_RANDOMFACTORY_H
#define ALPS_RANDOMFACTORY_H

#include <alps/random/random.h>
#include <alps/factory.h>
#include <string>

namespace alps {

class RNGFactory : public factory<std::string,BufferedRandomNumberGeneratorBase>
{
public:
  RNGFactory();
  template <class RNG> void register_rng(const std::string& name) 
  { register_type<BufferedRandomNumberGenerator<RNG> >(name);}
};

extern RNGFactory rng_factory;

} // end namespace

#endif // ALPS_RANDOMFACTORY_H
