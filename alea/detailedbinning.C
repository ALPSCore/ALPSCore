/***************************************************************************
* ALPS++/alea library
*
* test/alea/detailedbinning.C     simple example program
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include <iostream>
#include <alps/alea.h>
#include <alps/parameters.h> 
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
  alps::RealObservable obs_a("observable a");
  alps::RealObservable obs_b("observable b");

  //READ PARAMETERS
  //---------------
  alps::Parameters parms(std::cin);
  uint32_t thermalization_steps=parms.value_or_default("THERMALIZATION",1000);
  uint32_t number_of_steps=parms.value_or_default("STEPS",10000);

  //ADD MEASUREMENTS TO THE OBSERVABLES
  //----------------------------------- 
  for(uint i = 0; i < thermalization_steps; ++i){ 
    obs_a << random();
    obs_b << random()+1;
  }

  //RESET OBSERVABLES (THERMALIZATION FINISHED)
  //-------------------------------------------
  obs_a.reset(true);
  obs_b.reset(true);

  //ADD MEASUREMENTS TO THE OBSERVABLES
  //-----------------------------------
  for(uint32_t i = 0; i < number_of_steps; ++i){
    obs_a << random();
    obs_b << random()+1;
  }

  //OUTPUT OBSERVABLES
  //---------------------
  std::cout << "\n";
  std::cout << obs_a;       
  std::cout << "\n";
  std::cout << obs_b;
  std::cout << "\n";

  //JACKKNIVE ANALYSIS
  //------------------
  alps::RealObsevaluator obseval_a(obs_a);
  alps::RealObsevaluator obseval_b(obs_b);
  alps::RealObsevaluator obseval_c;
  obseval_c = obseval_b / obseval_a;
  std::cout << obseval_c;

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
}
