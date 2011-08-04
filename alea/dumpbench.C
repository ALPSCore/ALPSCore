/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Lukas Gamper <gamperl -at- gmail.com>
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

#include <alps/alea.h>
#include <alps/osiris/xdrdump.h> 
#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

#include <string>
#include <iostream>

int main() {
  int count = 100;
  double start = 0;
  
  std::string const xdr_filename = "test.dump";
  std::string const hdf5_filename = "test.h5";
  if (boost::filesystem::exists(boost::filesystem::path(xdr_filename)))
    boost::filesystem::remove(boost::filesystem::path(xdr_filename));
  if (boost::filesystem::exists(boost::filesystem::path(hdf5_filename)))
    boost::filesystem::remove(boost::filesystem::path(hdf5_filename));

  alps::ObservableSet measurement;
  {
    boost::timer t;
    boost::minstd_rand0 engine;
    boost::uniform_01<boost::minstd_rand0> random(engine);
    measurement << alps::make_observable(alps::SimpleRealObservable("Test"), true)
                << alps::RealObservable("Sign")
                << alps::RealObservable("No Measurements")
                << alps::IntHistogramObservable("Histogram", 0, 10)
                << alps::RealObservable("Test 2")
                << alps::RealObservable("Test 3")
//                << alps::RealVectorObservable("Test 4")
    ;
    std::valarray<double> vec;
    vec.resize(1000);
    for (int i = 0; i < 1000000; ++i) {
      vec[i % vec.size()] = random();
      measurement["Test"] << random();
      measurement["Sign"] << 1.0;
      measurement["Histogram"] << static_cast<int>(10*random());
      measurement["Test 2"] << random();
      measurement["Test 3"] << random();
//      measurement["Test 4"] << vec;
    }
    alps::RealObsevaluator e2 = measurement["Test 2"];
    alps::RealObsevaluator e4 = measurement["Test 3"];
    alps::RealObsevaluator ratio("Ratio");
    ratio = e2 / e4;
    measurement.addObservable(ratio);
    std::cerr << "Generating mesurement: " << t.elapsed() << " sec\n";
  }
  
  {
    boost::timer t;
    alps::hdf5::archive oar(hdf5_filename, alps::hdf5::archive::WRITE);
    for (int c = 0; c < count; ++c) {
      oar << make_pvp("/test/" + boost::lexical_cast<std::string>(c) + "/result", measurement);
    }
    std::cerr << "Writing to HDF5: " << t.elapsed() << " sec\n";
  }

  {
    boost::timer t;
    alps::OXDRFileDump dump(boost::filesystem::path(xdr_filename, boost::filesystem::native));
    for (int c = 0; c < count; ++c) {
      dump << measurement;
    }
    std::cerr << "Writing to XDR: " << t.elapsed()<< " sec\n";
  }

  measurement.clear();
  {
    boost::timer t;
    alps::hdf5::archive iar(hdf5_filename, alps::hdf5::archive::READ);
    for (int c = 0; c < count; ++c) {
      iar >> make_pvp("/test/" + boost::lexical_cast<std::string>(c) + "/result", measurement);
    }
    std::cerr << "Reading from HDF5: " << t.elapsed() << " sec\n";
  }

  measurement.clear();
  {
    boost::timer t;
    alps::IXDRFileDump dump(boost::filesystem::path(xdr_filename, boost::filesystem::native));
    for (int c = 0; c < count; ++c) {
      dump >> measurement;
    }
    std::cerr << "Reading from XDR: " << t.elapsed() << " sec\n";
  }

  boost::filesystem::remove(boost::filesystem::path(hdf5_filename));
  boost::filesystem::remove(boost::filesystem::path(xdr_filename));
}
