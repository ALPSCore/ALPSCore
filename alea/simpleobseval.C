/***************************************************************************
* PALM++/alea library
*
* alea/simpleobseval.C    simple example program for observable evaluator
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/alea.h>
#include <boost/random.hpp> 
#include <iostream>
#include <iomanip>

int main()
{
#ifndef BOOST_NO_EXCEPTIONS
try {
#endif
  // std::cout << std::setprecision(10);

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
  alps::RealObservable obs_c("observable c");
  obs_a.reset(true);
  obs_b.reset(true);
  obs_c.reset(true);

  for(int i=0; i < (1<<12); ++i) {
    obs_a << random();
    obs_b << random()+1;
    obs_c << random()+1 << random()+1 << random()+1;
  }

  std::cout << obs_a;
  std::cout << obs_b;
  std::cout << obs_c;

  //JACKKNIVE ANALYSIS
  //------------------

  std::cout << "Jackknife analysis\n";

  alps::RealObsevaluator obseval_a(obs_a);
  std::cout << obseval_a;
  std::cout << "  count = " << obseval_a.count()
	    << ", bin size = " << obseval_a.bin_size()
	    << ", number of bins = " << obseval_a.bin_number()
	    << std::endl;

  alps::RealObsevaluator obseval_b(obs_b);
  std::cout << obseval_b;
  std::cout << "  count = " << obseval_b.count()
	    << ", bin size = " << obseval_b.bin_size()
	    << ", number of bins = " << obseval_b.bin_number()
	    << std::endl;
  alps::RealObsevaluator obseval_c(obs_c);
  std::cout << obseval_c;
  std::cout << "  count = " << obseval_c.count()
	    << ", bin size = " << obseval_c.bin_size()
	    << ", number of bins = " << obseval_c.bin_number()
	    << std::endl;

  std::cout << "five different methods to construct RealObsevaluator\n";

  alps::RealObsevaluator obseval_0 = obseval_b / obseval_a;
  std::cout << "  " << obseval_0;

  alps::RealObsevaluator obseval_1(obseval_b / obseval_a);
  std::cout << "  " << obseval_1;

  alps::RealObsevaluator obseval_2(obseval_b / obseval_a, "obseval_e");
  std::cout << "  " << obseval_2;

  alps::RealObsevaluator obseval_3("obseval_f");
  obseval_3 = (obseval_b / obseval_a);
  std::cout << "  " << obseval_3;

  alps::RealObsevaluator obseval_4("obseval_g");
  obseval_4 << (obseval_b / obseval_a);
  std::cout << "  " << obseval_4;

  std::cout << "  count = " << obseval_4.count()
	    << ", bin size = " << obseval_4.bin_size()
	    << ", number of bins = " << obseval_4.bin_number()
	    << std::endl;

  std::cout << "merging observables b and c\n";
  alps::RealObsevaluator obseval_5("obseval_5");
  obseval_5 << obseval_b << obseval_c;
  std::cout << "  " << obseval_5;
  std::cout << "  count = " << obseval_5.count()
	    << ", bin size = " << obseval_5.bin_size()
	    << ", number of bins = " << obseval_5.bin_number()
	    << std::endl;
  
  alps::RealObsevaluator obseval_h = 1.0 / obseval_a;
  std::cout << obseval_h;

  alps::RealObsevaluator obseval_i = 1.0 / obseval_b;
  std::cout << obseval_i;

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

