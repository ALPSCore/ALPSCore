/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

void count_test_scalar(alps::accumulators::accumulator_set & measurements, std::string name) {
	std::string name1 = name + "m1";
	std::string name2 = name + "m2";
	for (int i = 0; i < 1000; ++i) {
		measurements[name1] << (i % 2) + 1;
	}
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);

    EXPECT_TRUE(results[name1].count() > 0);
    EXPECT_TRUE(results[name2].count() == 0);
}

void count_test_vector(alps::accumulators::accumulator_set & measurements, std::string name) {
	std::string name1 = name + "v1";
	std::string name2 = name + "v2";
        int L=10;
	for (int i = 0; i < 1000; ++i) {
		measurements[name1] << std::vector<double>(L, (i % 2) + 1);
	}
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);

    EXPECT_TRUE(results[name1].count() > 0);
    EXPECT_TRUE(results[name2].count() == 0);
}

#define ALPS_TEST_RUN_MUL_CONST_TEST(type, name)												\
	TEST(accumulators, divide_accumulators_scalar ## name) {										\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: type <double>( #name "m1") 					\
					 << alps::accumulators:: type <double>( #name "m2");					\
		count_test_scalar(measurements, #name);										\
	}																							\
	TEST(accumulators, divide_accumulators_vector ## name) {										\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: type <std::vector<double> >( #name "v1")	\
					 << alps::accumulators:: type <std::vector<double> >( #name "v2");	\
		count_test_vector(measurements, #name);										\
	}

ALPS_TEST_RUN_MUL_CONST_TEST(MeanAccumulator, meanDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(NoBinningAccumulator, noBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(LogBinningAccumulator, logBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(FullBinningAccumulator, fullBinningDouble)
#undef ALPS_TEST_RUN_MUL_CONST_TEST

