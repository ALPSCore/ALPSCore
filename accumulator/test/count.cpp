/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.h>
#include <alps/accumulator/accumulator.hpp>
#include "gtest/gtest.h"

TEST(accumulator, count_feature){

	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealObservable("scalar")
				 << alps::accumulator::RealVectorObservable("vector");

	for (int i = 1; i < 1001; ++i) {
		measurements["scalar"] << i;
		EXPECT_EQ(count(measurements["scalar"]) , i);
		measurements["vector"] << std::vector<double>(10, i);
		EXPECT_EQ(count(measurements["vector"]) , i);
	}

	alps::accumulator::result_set results(measurements);
	EXPECT_EQ(count(results["scalar"]) , 1000);
	EXPECT_EQ(count(results["vector"]) , 1000);
}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

