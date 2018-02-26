/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

TEST(accumulators, WeightedObservable){
	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::FullBinningAccumulator<double>("sign")
				 << alps::accumulators::FullBinningAccumulator<double>("x*sign")
				 << alps::accumulators::FullBinningAccumulator<std::vector<double> >("sign vec")
				 << alps::accumulators::FullBinningAccumulator<std::vector<double> >("x*sign vec");

        int s=4;
	for (int i = 1; i < 1000; ++i) {
		double sign = i % 3 ? 1. : -1.;
		measurements["sign"] << sign;
		measurements["sign vec"] << std::vector<double>(s, sign);
		measurements["x*sign"] << sign * i;
		measurements["x*sign vec"] << std::vector<double>(s, sign * i);
	}
	EXPECT_NEAR(measurements["sign"].mean<double>(), 1. / 3., 1.e-12);
	EXPECT_NEAR(measurements["x*sign"].mean<double>(), 166., 1.e-12);
        
	std::vector<double> x_sign_vec_mean=measurements["x*sign vec"].mean<std::vector<double> >();
        for(int i=0;i<s;++i){
          EXPECT_NEAR(x_sign_vec_mean[i], 166., 1.e-12);
        }

	alps::accumulators::result_set results(measurements);
	EXPECT_NEAR(results["sign"].mean<double>(), 1. / 3., 1.E-12);
	EXPECT_NEAR(results["x*sign"].mean<double>(), 166., 1E-12);
        // FIXME: test rather than print
	std::cout << (results["x*sign"] / results["sign"]) << std::endl;
        std::cout << (results["x*sign vec"] / results["sign"]) << std::endl;
	std::cout << (results["x*sign vec"] / results["sign vec"]) << std::endl;
}

// FIXME: this should go to a different test file
TEST(accumulators, BinaryWithScalar)
{
  alps::accumulators::accumulator_set measurements;
  typedef std::vector<double> double_vec;
  typedef std::vector<float> float_vec;
  measurements << alps::accumulators::FullBinningAccumulator<double>("double_scalar1")
               << alps::accumulators::FullBinningAccumulator<double>("double_scalar2")
               << alps::accumulators::FullBinningAccumulator<double_vec>("double_vector1")
               << alps::accumulators::FullBinningAccumulator<double_vec>("double_vector2")
               << alps::accumulators::FullBinningAccumulator<float>("float_scalar1")
               << alps::accumulators::FullBinningAccumulator<float>("float_scalar2")
               << alps::accumulators::FullBinningAccumulator<float_vec>("float_vector1")
               << alps::accumulators::FullBinningAccumulator<float_vec>("float_vector2");

  measurements["double_scalar1"] << 1.;
  measurements["double_scalar2"] << 2.;
  measurements["float_scalar1"] << 1.;
  measurements["float_scalar2"] << 2.;

  measurements["double_vector1"] << double_vec(3,1.);
  measurements["double_vector2"] << double_vec(3,2.);
  measurements["float_vector1"] << float_vec(3,1.);
  measurements["float_vector1"] << float_vec(3,2.);

  alps::accumulators::result_set results(measurements);

  results["double_scalar1"]/results["double_scalar2"]; // FIXME: duplicate of other tests
  EXPECT_THROW(results["double_scalar1"]/results["float_scalar2"],std::logic_error);
  EXPECT_THROW(results["double_vector1"]/results["float_vector2"],std::logic_error);
  
  results["double_vector1"]/results["double_vector2"]; // FIXME: duplicate of other tests
  EXPECT_THROW(results["double_scalar1"]/results["double_vector2"],std::logic_error);

  results["double_vector1"]/results["double_scalar2"]; 
  EXPECT_THROW(results["double_vector1"]/results["float_scalar2"],std::logic_error);
}
