/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
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
	EXPECT_EQ(results["sign"].mean<double>(), 1. / 3.);
	EXPECT_EQ(results["x*sign"].mean<double>(), 166.);
	std::cout << (results["x*sign"] / results["sign"]) << std::endl;
        //we really need to fix this so we can divide vectors by scalars
	EXPECT_ANY_THROW(std::cout << (results["x*sign vec"] / results["sign"]) << std::endl);
	std::cout << (results["x*sign vec"] / results["sign vec"]) << std::endl;
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
