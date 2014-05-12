/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2008 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parapack/wanglandau.h>
#include <alps/alea.h>
#include <boost/random.hpp>
#include <iostream>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  boost::mt19937 eng;
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
    rng(eng, boost::uniform_real<>());

  alps::Parameters params;
  params["DIR_NAME"] = ".";

  alps::ObservableSet obs0;
  alps::integer_range<int> range0(-10, 13);
  alps::wanglandau_weight weight0(range0);
  weight0.init_observables(obs0);
  obs0.reset(true);
  for (int i = 0; i < 100; ++i)
    weight0.visit(obs0, static_cast<int>(range0.size() * rng()) + range0.min(), 2);
  weight0.write_observables(obs0);
  params["WEIGHT_DUMP_FILE"] = "wl_weight.dump0";
  alps::wanglandau_weight::save_weight(obs0, params);
  std::cout << "# weight0:\n";
  for (int b = range0.min(); b <= range0.max(); ++b)
    std::cout << b << ' ' << log(weight0[b]) << std::endl;

  alps::ObservableSet obs1;
  alps::integer_range<int> range1(10, 30);
  alps::wanglandau_weight weight1(range1);
  weight1.init_observables(obs1);
  obs1.reset(true);
  for (int i = 0; i < 1000; ++i)
    weight1.visit(obs1, static_cast<int>(range1.size() * rng()) + range1.min(), 2);
  weight1.write_observables(obs1);
  params["WEIGHT_DUMP_FILE"] = "wl_weight.dump1";
  alps::wanglandau_weight::save_weight(obs1, params);
  std::cout << "# weight1:\n";
  for (int b = range1.min(); b <= range1.max(); ++b)
    std::cout << b << ' ' << log(weight1[b]) << std::endl;

  alps::integer_range<int> range2 = unify(range0, range1);
  alps::wanglandau_weight weight2(range2);
  params["WEIGHT_DUMP_FILE"] = "wl_weight.dump0 wl_weight.dump1";
  weight2.load_weight(params);
  std::cout << "# weight2:\n";
  for (int b = range2.min(); b <= range2.max(); ++b)
    std::cout << b << ' ' << log(weight2[b]) << std::endl;

#ifndef BOOST_NO_EXCEPTIONS
}
catch (std::exception& exp) {
  std::cerr << exp.what() << std::endl;
  std::abort();
}
#endif
  return 0;
}
