/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2007 by Synge Todo <wistaria@comp-phys.org>
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

/* $Id: vectorobseval.C 2190 2006-08-30 09:28:03Z wistaria $ */

#include <alps/alea.h>
#include <boost/random.hpp> 

int main() {
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  typedef boost::minstd_rand0 random_base_type;
  typedef boost::uniform_01<random_base_type> random_type; 
  random_base_type random_int;
  random_type random(random_int); 

  alps::ObservableSet obs1;
  obs1 << alps::SimpleRealObservable("scalar 1");
  obs1 << alps::HistogramObservable<alps::int32_t>("histogram 1", 0, 10);
  obs1 << alps::HistogramObservable<double>("histogram 2", 0, 1, 0.1);

  for (int i=0; i < (1<<12); ++i) {
    double r = random();
    obs1["scalar 1"] << r*r;
	obs1["histogram 1"] << static_cast<alps::int32_t>(10*r);
    obs1["histogram 2"] << r*r;
  }
  
  alps::oxstream oxs;
  obs1.write_xml(oxs);

  alps::ObservableSet obs2;
  obs2 << alps::SimpleRealObservable("scalar 1");
  obs2 << alps::HistogramObservable<alps::int32_t>("histogram 1", 0, 10);
  obs2 << alps::HistogramObservable<double>("histogram 2", 0, 1, 0.1);

  for (int i=0; i < (1<<10); ++i) {
    double r = random();
    obs2["scalar 1"] << r*r;
	obs2["histogram 1"] << static_cast<alps::int32_t>(10*r);
    obs2["histogram 2"] << r*r;
  }
  
  obs2.write_xml(oxs);

  obs1 << obs2;
  obs1.write_xml(oxs);

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
