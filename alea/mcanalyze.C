/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* Copyright (C) 2011-2012 by Lukas Gamper <gamperl@gmail.com>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Maximilian Poprawe <poprawem@ethz.ch>
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
#include <alps/alea/mcanalyze.hpp>

#include <alps/utility/encode.hpp>
#include <alps/utility/size.hpp>

#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>
#include <boost/random.hpp>

#include <iostream>
#include <vector>
#include <iomanip>
#include <string>
#include <algorithm>


int main() {
  using alps::size;
  try {
    std::string const filename = "test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    {
        boost::minstd_rand0 engine;
        boost::uniform_01<boost::minstd_rand0> random(engine);
        alps::ObservableSet measurement;
        measurement << alps::RealObservable("Scalar")
                    << alps::RealVectorObservable("Vector");

        double tmp(random());
        for (int i = 0; i < 10000; ++i) {
            measurement["Scalar"] << 0.9 * tmp + random();
            //measurement["Vector"] << 1.0;
        }
        
        alps::hdf5::archive oar(filename, "a");
        oar << make_pvp("/test/result", measurement);
    }
    {
        alps::ObservableSet measurement;

        alps::alea::mcdata<double> scalar_data;
        scalar_data.load(filename, "/test/result/Scalar");
        std::cout << scalar_data;

        std::cout << size(scalar_data) << "\n";
        std::cout << alps::alea::mean(scalar_data) << "\n";





    }
    boost::filesystem::remove(boost::filesystem::path(filename));
  }
  catch (std::exception& e) {
    std::cerr << "Fatal error: " << e.what() << "\n"; 
  }
  return 0;
}








/*

  std::cout << "\n************** MCANALYZE C++ TEST ***************\n";
  std::cout << "\nErrors saved in file before manipulation:\n";

  const std::string filename = "mcanalyze.input";
  std::vector<alps::alea::mcdata<double> > mcdata_objects;
  std::vector<std::string> var_names;

{ 
  alps::hdf5::iarchive iar(filename);
  var_names = iar.list_children("/simulation/results");
  for (std::vector<std::string>::const_iterator it = var_names.begin(); it != var_names.end(); ++it) {
      iar.set_context("/simulation/results/" + iar.encode_segment(*it));
      if (iar.is_scalar("/simulation/results/" + iar.encode_segment(*it) + "/mean/value")) {
          alps::alea::mcdata<double> obs;
          //obs.serialize(iar);
          mcdata_objects.push_back(obs);
          std::cout << "  " << *it << " " << obs.error() << std::endl;
      } else {
          alps::alea::mcdata<std::vector<double> > obs;
          //obs.serialize(iar);
          std::cout << "  " << *it << " " << obs.error() << std::endl;
      }
  }
}  

  using namespace alps::alea;
{
  alps::hdf5::oarchive oar(filename);
*/
/*
    All many functions have a _range and a _decay version.
    The _range function expects size_t argument(s) and one gives the actual indeces.
      example1: The first 5 autocorrelation terms (lags) :                      autocorrelation_range (timeseries, 5 ) 
      example2: A fit of the autocorrelation from the 1st to the 7th lag:       exponential_autocorrelation_time_range (autocorrelation, 1, 7)

    The _decay function expects double argument/s which is/are the percentage to which the user wants the timeseries to fall to. the function then calls the _range version with the appropriate values
      example1: The autocorrelation calculated until it decayed to 0.5% of its initial value:         autocorrelation_decay (timeseries, 0.005)
      example2: A fit of the autocorrelation when it is between 90% and 10% of its initial value:     exponential_autocorrelation_time_decay (autocorrelation, 0.9, 0.1)
*/



// Observable 0 in file
// calculate and write to file the error of the mean with a binning analysis
//  oar << alps::make_pvp("/simulation/results/" + oar.encode_segment(var_names[0]) + "/mean/error", error(mcdata_objects[0], binning));


// Observable 1 in file
// Autocorrelation
//  mctimeseries<double> autocorrelation;
//  autocorrelation.shallow_assign( autocorrelation_decay (mcdata_objects[1], 0.1 )); // shallow_assign - else a copy of the data would be made
                                                                    // maybe give the autocorrelation function a mctimeseries object to write to? like the lapack functions?
                                                                    // or do we not care about an extra copy and just write:
                                                                    // mctimeseries<double> autocorrelation( autocorrelation_decay (mcdata_objects[1], 0.1 ));     ?

// Exponential fit of autocorrelation
//  exponential<double> exponential_autocorrelation_time = exponential_autocorrelation_time_decay (autocorrelation, 1, 0.2);

// sum up and integrate fit to get integrated autocorrelation time
//  double integrated_autocorrelation_time = integrated_autocorrelation_time_decay (autocorrelation, exponential_autocorrelation_time, 0.2);

// calculate and write the error with autocorrelation-correction 
//  oar << alps::make_pvp("/simulation/results/" + oar.encode_segment(var_names[1]) + "/mean/error", error(mcdata_objects[1], uncorrelated) * std::sqrt(1 + 2 * integrated_autocorrelation_time));


// Observable 2 in file
// calculate and write the error to file assuming uncorrelated data
/*  oar << alps::make_pvp("/simulation/results/" + oar.encode_segment(var_names[2]) + "/mean/error", error(mcdata_objects[2], uncorrelated_selector()));

  } 
  {
    std::cout << "\nData saved in file after manipulation:\n";
    alps::hdf5::iarchive iar(filename);

    for (std::vector<std::string>::const_iterator it = var_names.begin(); it != var_names.end(); ++it) {
        iar.set_context("/simulation/results/" + iar.encode_segment(*it));
        if (iar.is_scalar("/simulation/results/" + iar.encode_segment(*it) + "/mean/value")) {
          alps::alea::mcdata<double> obs;
          //obs.serialize(iar);
          std::cout << "  " << *it << " " << obs.error() << std::endl;
        } else {
          alps::alea::mcdata<std::vector<double> > obs;
          //obs.serialize(iar);
          std::cout << "  " << *it << " " << obs.error() << std::endl;
        }
    }
  }

    std::cout << "\n************** TEST END ***************\n";

    return 0;
  }

*/
/*

  typedef boost::mt19937 engine_type;
  typedef boost::uniform_real<double> dist_type;
  typedef boost::variate_generator<boost::mt19937&,dist_type> rng_type;


  engine_type eng(42);
  rng_type rng(eng,dist_type(-1,1));


  std::valarray<double> tmp(1.,3);

  alps::RealVectorObservable obs("vecobservable");

  for( int i = 0; i < 1000000; ++i )
  {
    for (int j = 0; j<3; ++j) {
      tmp[j] = 0.9 * tmp[j] + 0.1 * rng();
    }
    obs << tmp;
  }  
alps::alea::mcdata<std::vector<double> > obseval(obs);

*/
/*
  double tmp = 1;
  alps::RealObservable obs("observable");

  for( int i = 0; i < 100; ++i )
  {
    tmp = 0.9 * tmp + rng();
    obs << tmp;
  }

*/

