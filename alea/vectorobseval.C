/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006-2010 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/alea.h>
#include <boost/random.hpp>
#include <iostream>
#include <iomanip>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  typedef boost::minstd_rand0 random_base_type;
  typedef boost::uniform_01<random_base_type> random_type;
  random_base_type random_int;
  random_type random(random_int);

  alps::RealVectorObservable obs_a("observable");

  for(int i=0; i < (1<<12); ++i) {
    std::valarray<double> obs(2);
    obs[0] = random();
    obs[1] = random() + 1;
    obs_a << obs;
  }
  std::cout << obs_a;

  alps::RealVectorObsevaluator veceval0(obs_a);
  alps::RealVectorObsevaluator veceval1(obs_a);
  std::valarray<double> vec(2);
  vec[0] = 3;
  vec[1] = -2;

  std::cout << veceval0;

  std::cout << veceval0 + 1.0;
  std::cout << 2.5 + veceval0;
  std::cout << veceval0 + veceval1;
  std::cout << vec + veceval0;

  std::cout << veceval0 - 1.0;
  std::cout << 2.5 - veceval0;
  std::cout << veceval0 - veceval1;
  std::cout << veceval0 - vec;

  std::cout << veceval0 * 3.0;
  std::cout << 1.3 * veceval0;
  std::cout << veceval0 * veceval1;
  std::cout << veceval0 * vec;
  std::cout << vec * veceval0;

  std::cout << veceval0 / 3.0;
  std::cout << 1.3 / veceval0;
  std::cout << veceval0 / veceval1;
  std::cout << veceval0 / vec;
  std::cout << vec / veceval0;

  std::cout << pow(veceval0, 2);

  alps::RealObsevaluator eval0(obs_a.slice(0));
  alps::RealObsevaluator eval1(obs_a.slice(1));
  std::cout << eval0 << eval1 << eval0 * eval1;

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exc) {
  std::cerr << exc.what() << "\n";
  return -1;
}
catch (...) {
  std::cerr << "Fatal Error: Unknown Exception!\n";
  return -2;
}
#endif
  return 0;
}

