/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parapack/exchange.h>
#include <algorithm>
#include <iostream>
#include <boost/random.hpp>

int main() {
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  typedef alps::parapack::exmc::inverse_temperature_set inverse_temperature_set;

  boost::mt19937 eng;
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
    rng(eng, boost::uniform_real<>());

  std::cout << std::setprecision(3);

  int n = 10;

  std::vector<double> x, y;
  for (int i = 0; i < n; ++i) {
    x.push_back(rng());
    y.push_back(rng());
  }
  std::sort(x.begin(), x.end());
  std::sort(y.begin(), y.end());
  for (int i = 0; i < n; ++i) {
    std::cout << x[i] << '\t' << y[i] << std::endl;
  }


  double a, b;

  boost::tie(a,b) = inverse_temperature_set::linear_regression(10, 0, x, y);
  std::cout << "#\t" << 10 << '\t' << 0 << '\t' << a << '\t' << b << std::endl;

  boost::tie(a,b) = inverse_temperature_set::linear_regression(2, 2, x, y);
  std::cout << "#\t" << 2 << '\t' << 2 << '\t' << a << '\t' << b << std::endl;

  boost::tie(a,b) = inverse_temperature_set::linear_regression(5, 3, x, y);
  std::cout << "#\t" << 5 << '\t' << 3 << '\t' << a << '\t' << b << std::endl;

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
