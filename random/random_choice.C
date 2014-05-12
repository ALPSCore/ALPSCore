/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006-2013 by Synge Todo <wistaria@comp-phys.org>
*
* This software is published under the ALPS Application License; you
* can use, redistribute it and/or modify it under the terms of the
* license, either version 1 or (at your option) any later version.
* 
* You should have received a copy of the ALPS Application License
* along with this software; see the file LICENSE. If not, the license
* is also available from http://alps.comp-phys.org/.
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

#include <alps/random/random_choice.hpp>
#include <boost/random.hpp>
#include <cmath>
#include <vector>

#define BOOST_TEST_MODULE random_choice
#ifndef BOOST_TEST_DYN_LINK
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif

static const unsigned int n = 9;
static const unsigned int samples = 100000;

BOOST_AUTO_TEST_CASE(random_choice) {
  std::cout << "number of bins = " << n << std::endl;
  std::cout << "number of samples = " << samples << std::endl;

  std::vector<double> weights(n);

  // random number generator
  typedef boost::mt19937 engine_type;
  typedef boost::variate_generator<boost::mt19937&, boost::uniform_real<> > generator_type;
  engine_type eng(29411);
  generator_type rng(eng, boost::uniform_real<>());

  // generate weights
  std::generate(weights.begin(), weights.end(), rng);
  double tw = 0;
  for (unsigned int i = 0; i < n; ++i) tw += weights[i];

  // double-base version
  {
    alps::random_choice<generator_type> dist(weights);
    BOOST_CHECK(dist.check(weights));

    std::vector<double> accum(n, 0);
    for (unsigned int t = 0; t < samples; ++t) ++accum[dist(rng)];

    std::cout << "bin\tweight\t\tresult\t\tdiff\t\tsigma\t\tdiff/sigma\n";
    for (unsigned int i = 0; i < n; ++i) {
      double diff = std::abs((weights[i] / tw) - (accum[i] / samples));
      double sigma = std::sqrt(accum[i]) / samples;
      std::cout << i << "\t" << (weights[i] / tw) << "    \t"
                << (accum[i] / samples) << "    \t" << diff << "    \t"
                << sigma << "    \t" << (diff / sigma) << std::endl;
      BOOST_CHECK(diff < 3 * sigma);
    }
  }

  // integer-base version
  {
    alps::random_choice<engine_type> dist(weights);
    BOOST_CHECK(dist.check(weights));

    std::vector<double> accum(n, 0);
    for (unsigned int t = 0; t < samples; ++t) ++accum[dist(eng)];

    std::cout << "bin\tweight\t\tresult\t\tdiff\t\tsigma\t\tdiff/sigma\n";
    for (unsigned int i = 0; i < n; ++i) {
      double diff = std::abs((weights[i] / tw) - (accum[i] / samples));
      double sigma = std::sqrt(accum[i]) / samples;
      std::cout << i << "\t" << (weights[i] / tw) << "    \t"
                << (accum[i] / samples) << "    \t" << diff << "    \t"
                << sigma << "    \t" << (diff / sigma) << std::endl;
      BOOST_CHECK(diff < 3 * sigma);
    }
  }
}
