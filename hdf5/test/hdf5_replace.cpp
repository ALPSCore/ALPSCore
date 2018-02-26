/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>

#include <vector>
#include "gtest/gtest.h"


using namespace std;

TEST(hdf5, TestingReplace){

   vector<double> vec(100, 10.);

   {
       alps::hdf5::archive ar("res.h5", "a");
       ar << alps::make_pvp("/vec2", vec);
   }
   {
       alps::hdf5::archive ar("res.h5", "w");
       ar << alps::make_pvp("/vec", vec);
   }
   {
       vector<double> tmp;
       alps::hdf5::archive ar("res.h5");
       ar >> alps::make_pvp("/vec2", tmp);
   }

}
