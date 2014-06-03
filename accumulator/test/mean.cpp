/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.h>
#include <alps/accumulator/accumulator.hpp>
#include "gtest/gtest.h"


TEST(accumulator, mean_feature){

	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealObservable("obs1")
				 << alps::accumulator::RealObservable("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << 1.;
		EXPECT_EQ(measurements["obs1"].mean<double>() , 1.);
		measurements["obs2"] << i;
		EXPECT_EQ(measurements["obs2"].mean<double>() , double(i + 1) / 2.);
	}

	alps::accumulator::result_set results(measurements);
	EXPECT_EQ(results["obs1"].mean<double>() , 1.);
	EXPECT_EQ(results["obs2"].mean<double>() , 500.);
}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

