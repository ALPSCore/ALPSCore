/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2009 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parapack/temperature_scan.h>
#include <iomanip>
#include <iostream>

class my_worker {
public:
  my_worker(alps::Parameters const&) {}
  void init_observables(alps::Parameters const&, alps::ObservableSet const&) {}
  void run(alps::ObservableSet const&) {}
  void set_beta(double beta) { std::cout << "T = " << 1/beta; }
  void save(alps::ODump&) const {}
  void load(alps::IDump&) {}
};

int main() {
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif
  std::cout << std::setprecision(3);

  alps::Parameters params(std::cin);
  std::vector<alps::ObservableSet> obs;
  alps::parapack::temperature_scan_adaptor<my_worker> worker(params);
  worker.init_observables(params, obs);

  int count = 0;
  while (worker.progress() < 1) {
    std::cout << count++ << ", ";
    worker.run(obs);
    std::cout << ", progress = " << worker.progress() << std::endl;
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
