/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

/* $Id$ */

#include <iostream>
#include <alps/alea.h>
#include <alps/parameter.h> 
#include <alps/osiris/xdrdump.h> 
#include <boost/filesystem/operations.hpp>
#include <boost/random.hpp> 

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif

  //DEFINE RANDOM NUMBER GENERATOR
  //------------------------------
  typedef boost::minstd_rand0 random_base_type;
  typedef boost::uniform_01<random_base_type> random_type;
  random_base_type random_int;
  random_type random(random_int);

  //DEFINE OBSERVABLES
  //------------------
  alps::ObservableSet measurement;
  measurement << alps::RealObservable("Sign");
  measurement << alps::SignedObservable<alps::RealObservable>("Observable a");
  measurement << alps::SignedObservable<alps::RealObservable>("Observable b");

  //READ PARAMETERS
  //---------------
  alps::Parameters parms(std::cin);
  uint32_t thermalization_steps=parms.value_or_default("THERMALIZATION",1000);
  uint32_t number_of_steps=parms.value_or_default("STEPS",10000);

  // THERMALIZATION
  //----------------------------------- 
  for(uint32_t i = 0; i < thermalization_steps; ++i){ 
    random();
    random();
    random();
  }


  //ADD MEASUREMENTS TO THE OBSERVABLES
  //-----------------------------------
  for(uint32_t i = 0; i < number_of_steps; ++i){
    double sign = (random() < 0.4 ? -1. : 1.);
    measurement["Sign"] << sign;
    measurement["Observable a"] << sign*random();
    measurement["Observable b"] << sign*(random()+1.);
  }

  // SAVE and LOAD
  {
    alps::OXDRFileDump dump(boost::filesystem::path("Observableset.dump"));
    dump << measurement;
  }
  measurement.clear();
  {
    alps::IXDRFileDump dump(boost::filesystem::path("Observableset.dump"));
    dump >> measurement;
  }

  alps::RealObsevaluator obse_a = measurement["Observable a"];
  alps::RealObsevaluator obse_b = measurement["Observable b"];
  measurement << alps::RealObsevaluator("a/b");
  measurement.get<alps::RealObsevaluator>("a/b") = obse_a / obse_b;

  // SAVE and LOAD
  {
    alps::OXDRFileDump dump(boost::filesystem::path("Observableset.dump"));
    dump << measurement;
  }
  measurement.clear();
  {
    alps::IXDRFileDump dump(boost::filesystem::path("Observableset.dump"));
    dump >> measurement;
  }

  alps::oxstream oxs;
  measurement.write_xml(oxs);

  boost::filesystem::remove(boost::filesystem::path("Observableset.dump"));

  // test for signed observable with custom sign name
  measurement.clear();

  //DEFINE OBSERVABLES
  //------------------
  measurement << alps::RealObservable("My Sign");
  measurement << alps::SignedObservable<alps::RealObservable>("Observable c", "My Sign");

  //ADD MEASUREMENTS TO THE OBSERVABLES
  //-----------------------------------
  for(uint32_t i = 0; i < number_of_steps; ++i){
    double sign = (random() < 0.4 ? -1. : 1.);
    measurement["My Sign"] << sign;
    measurement["Observable c"] << sign*random();
  }
  measurement.write_xml(oxs);

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
