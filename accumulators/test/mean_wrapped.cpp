/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators_.hpp>
#include "gtest/gtest.h"

TEST(accumulator, mean_feature_scalar_MeanAccumulator_double) {

  	// mean_test_body_scalar<alps::MeanAccumulator<double>, double >();

	alps::accumulator_set measurements;
	measurements << alps::MeanAccumulator<double>("obs1") 
		<< alps::MeanAccumulator<double>("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << double(1);
		EXPECT_EQ(measurements["obs1"].count(), i);
		EXPECT_EQ(measurements["obs1"].mean<double>() , double(1));
		measurements["obs2"] << double(i);
		EXPECT_EQ(measurements["obs2"].count(), i);
		EXPECT_EQ(measurements["obs2"].mean<double>() , double(i + 1) / 2);
	}

	alps::result_set results(measurements);
	EXPECT_EQ(results["obs1"].mean<double>(), double(1));
	EXPECT_EQ(results["obs2"].mean<double>(), double(500));	
}

int main(int argc, char **argv) 
{
	::testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
