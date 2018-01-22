/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

TEST(accumulator, count_feature){

	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::MeanAccumulator<double>("scalar")
				 << alps::accumulators::MeanAccumulator<std::vector<double> >("vector");

	for (unsigned int i = 1; i < 1001; ++i) {
		measurements["scalar"] << i;
		EXPECT_EQ(count(measurements["scalar"]) , i);
		measurements["vector"] << std::vector<double>(10, i);
		EXPECT_EQ(count(measurements["vector"]) , i);
	}

	alps::accumulators::result_set results(measurements);
	EXPECT_EQ(count(results["scalar"]) , 1000u);
	EXPECT_EQ(count(results["vector"]) , 1000u);
}
