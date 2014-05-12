/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010-2012 by Lukas Gamper <gamperl -at- gmail.com>,
*                            Synge Todo <wistaria@comp-phys.org>
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

#include <alps/utility/encode.hpp>
#include <alps/alea.h>

#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
  try {
    std::string const filename = "observableset.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    {
        alps::ObservableSet measurement;
        measurement << alps::make_observable(alps::RealObservable("Test"), true)
                    << alps::RealObservable("Sign")
                    << alps::RealObservable("No Measurements")
                    << alps::IntHistogramObservable("Histogram", 0, 10)
                    << alps::RealObservable("Test 2")
                    << alps::RealObservable("Test 3")
                    << alps::SimpleRealObservable("Test 4");
        alps::hdf5::archive oar(filename, "a");
        oar["/test/0/result"] << measurement;
    }
    {
        boost::minstd_rand0 engine;
        boost::uniform_01<boost::minstd_rand0> random(engine);
        alps::ObservableSet measurement;
        alps::hdf5::archive iar(filename, "r");
        iar["/test/0/result"] >> measurement;
        for (int i = 0; i < 10000; ++i) {
          measurement["Test"] << random();
          measurement["Sign"] << 1.0;
          measurement["Histogram"] << static_cast<int>(10*random());
          measurement["Test 2"] << random();
          measurement["Test 3"] << random();
          measurement["Test 4"] << random();
        }
        alps::RealObsevaluator e2 = measurement["Test 2"];
        alps::RealObsevaluator e4 = measurement["Test 3"];
        alps::RealObsevaluator ratio("Ratio");
        ratio = e2 / e4;
        measurement.addObservable(ratio);
        
        alps::hdf5::archive oar(filename, "a");
        oar["/test/0/result"] << measurement;
        
        alps::IntHistogramObsevaluator eval = measurement["Histogram"];
    }
    {
        alps::ObservableSet measurement;
        measurement << alps::make_observable(alps::RealObservable("Test"), true)
                    << alps::RealObservable("Sign")
                    << alps::RealObservable("No Measurements")
                    << alps::IntHistogramObservable("Histogram", 0, 10);
        alps::hdf5::archive iar(filename, "r");
        iar["/test/0/result"] >> measurement;
        std::cout << measurement;
        alps::IntHistogramObsevaluator eval = measurement["Histogram"];
    }
    {
        alps::ObservableSet measurement;
        alps::hdf5::archive iar(filename, "r");
        iar["/test/0/result"] >> measurement;
        std::cout << measurement;
        alps::IntHistogramObsevaluator eval = measurement["Histogram"];
    }
    boost::filesystem::remove(boost::filesystem::path(filename));
  }
  catch (std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << "\n"; 
  }
  return 0;
}
