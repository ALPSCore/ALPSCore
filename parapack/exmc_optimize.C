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
#include <iomanip>
#include <iostream>
#include <boost/random.hpp>

struct dummy {
  typedef double weight_parameter_type;
  static double log_weight(weight_parameter_type gw, double beta) {
    return beta * gw;
  }
};

int main() {
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  std::cout << std::setprecision(4);

  int n = 10;

  boost::mt19937 eng;
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
    rng(eng, boost::uniform_real<>());

  alps::Parameters params;
  params["BETA_MAX"] = 5;
  params["BETA_MIN"] = 1;
  params["NUM_REPLICAS"] = n;

  alps::parapack::exmc::inverse_temperature_set beta_set(params);

  std::vector<double> beta(n);
  std::vector<dummy::weight_parameter_type> gw(n);
  for (int i = 0; i < n; ++i) {
    beta[i] = beta_set[i];
    gw[i] = rng();
  }
  std::sort(gw.begin(), gw.end());

  std::cout << "[before optimization]\n";
  std::cout << "beta\tenergy\tC\n";
  std::cout << beta_set[0] << '\t' << gw[0] << std::endl;
  for (int i = 1; i < n; ++i)
    std::cout << beta_set[i] << '\t' << gw[i] << '\t'
              << ((dummy::log_weight(gw[i-1], beta_set[i-1]) +
                   dummy::log_weight(gw[i], beta_set[i])) -
                  (dummy::log_weight(gw[i], beta_set[i-1]) +
                   dummy::log_weight(gw[i-1], beta_set[i])))
              << std::endl;

  std::cout << "Wg[2.33] = " << beta_set.interpolate<dummy>(beta, gw, 2.33) << std::endl;
  std::cout << "Wg[3.52] = " << beta_set.interpolate<dummy>(beta, gw, 3.52) << std::endl;

  beta_set.optimize_h1999<dummy>(gw);

  std::cout << "[after optimization]\n";
  std::cout << "beta\tenergy\tC\n";
  std::cout << beta_set[0] << '\t' << gw[0] << std::endl;
  for (int i = 1; i < n; ++i) {
    double w0 = beta_set.interpolate<dummy>(beta, gw, beta_set[i-1]);
    double w1 = beta_set.interpolate<dummy>(beta, gw, beta_set[i]);

    std::cout << beta_set[i] << '\t' << gw[i] << '\t'
              << ((dummy::log_weight(w0, beta_set[i-1]) +
                   dummy::log_weight(w1, beta_set[i])) -
                  (dummy::log_weight(w1, beta_set[i-1]) +
                   dummy::log_weight(w0, beta_set[i])))
              << std::endl;
  }

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
