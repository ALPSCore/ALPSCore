/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators/accumulator.hpp>
#include "gtest/gtest.h"

TEST(accumulators, WeightedObservable){
	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::FullBinningAccumulator<double>("sign")
				 << alps::accumulators::FullBinningAccumulator<double>("x*sign");

	for (int i = 1; i < 1000; ++i) {
		double sign = i % 3 ? 1. : -1.;
		measurements["sign"] << sign;
		measurements["x*sign"] << sign * i;
	}
	EXPECT_EQ(measurements["sign"].mean<double>(), 1. / 3.);
	EXPECT_EQ(measurements["x*sign"].mean<double>(), 166.);

	alps::accumulators::result_set results(measurements);
	EXPECT_EQ(results["sign"].mean<double>(), 1. / 3.);
	EXPECT_EQ(results["x*sign"].mean<double>(), 166.);
	std::cout << (results["x*sign"] / results["sign"]) << std::endl;
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
