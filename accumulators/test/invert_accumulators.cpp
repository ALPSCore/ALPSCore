/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"
//This program illustrates a problem with taking an inverse of an observable.

TEST(accumulators, invert_accumulators){
  srand48(0);

  alps::accumulators::accumulator_set measurements;
  measurements<<alps::accumulators::FullBinningAccumulator<double>("one_half");
  measurements<<alps::accumulators::FullBinningAccumulator<double>("one");
  int samples=1000;

  for(int count=0;count<samples;++count){
      measurements["one_half"]<<drand48();
      measurements["one"]<<1.;
  }

  alps::accumulators::result_set results(measurements);
  //we have measured 0.5:
  EXPECT_NEAR(results["one_half"].mean<double>(), 0.5, 1.e-1);


  //the inverse of 1/2 should be around two:
  EXPECT_NEAR((1./results["one_half"]).mean<double>(), 2.0, 1.e-1);

  //the inverse of 1/2 should be around two:
  EXPECT_NEAR((results["one"]/results["one_half"]).mean<double>(), 2.0, 1.e-1);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

