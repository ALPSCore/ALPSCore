/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators/accumulator.hpp>
#include "gtest/gtest.h"

TEST(accumulators, TestingMultiplyAccumulatorByConstant) {

	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::MeanAccumulator<double>("meanDouble")
		<< alps::accumulators::NoBinningAccumulator<double>("noBinningDouble")
		<< alps::accumulators::LogBinningAccumulator<double>("logBinningDouble")
		<< alps::accumulators::FullBinningAccumulator<double>("fullBinningDouble");

	for (int i = 0; i < 1000; ++i)
		measurements["fullBinningDouble"] << 2. * (i % 2);

	std::cout << measurements << std::endl;

	alps::accumulators::result_set results(measurements);

	std::cout << results["fullBinningDouble"] * 2 << std::endl;
	std::cout << 2 * results["fullBinningDouble"] << std::endl;

	std::cout << results["fullBinningDouble"] / 2 << std::endl;
	std::cout << 2 / results["fullBinningDouble"] << std::endl;
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

