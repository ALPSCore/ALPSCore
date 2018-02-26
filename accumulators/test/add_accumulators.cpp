/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

void test_add_accumulators_scalar(alps::accumulators::accumulator_set & measurements, std::string name) {
	std::string name1 = name + "_1_scalar";
	std::string name2 = name + "_2_scalar";
	for (int i = 0; i < 1000; ++i) {
		measurements[name1] << (i % 2) + 1;
		measurements[name2] << 2. * (i % 2);
	}
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);
	std::cout << results[name1] + results[name2] << std::endl;
	std::cout << results[name1] - results[name2] << std::endl;
}

void test_add_accumulators_vector(alps::accumulators::accumulator_set & measurements, std::string name) {
	std::string name1 = name + "_1_vector";
	std::string name2 = name + "_2_vector";
	for (int i = 0; i < 1000; ++i) {
		measurements[name1] << std::vector<double>(10, (i % 2) + 1);
		measurements[name2] << std::vector<double>(10, 2. * (i % 2));
	}
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);
	std::cout << results[name1] + results[name2] << std::endl;
	std::cout << results[name1] - results[name2] << std::endl;
}

#define ALPS_TEST_RUN_MUL_CONST_TEST(type, name)												\
	TEST(accumulators, add_accumulators_scalar ## name) {										\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: type <double>( #name "_1_scalar") 					\
					 << alps::accumulators:: type <double>( #name "_2_scalar");					\
		test_add_accumulators_scalar(measurements, #name);										\
	}																							\
	TEST(accumulators, add_accumulators_vector ## name) {										\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: type <std::vector<double> >( #name "_1_vector")	\
					 << alps::accumulators:: type <std::vector<double> >( #name "_2_vector");	\
		test_add_accumulators_vector(measurements, #name);										\
	}

ALPS_TEST_RUN_MUL_CONST_TEST(MeanAccumulator, meanDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(NoBinningAccumulator, noBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(LogBinningAccumulator, logBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(FullBinningAccumulator, fullBinningDouble)
#undef ALPS_TEST_RUN_MUL_CONST_TEST

