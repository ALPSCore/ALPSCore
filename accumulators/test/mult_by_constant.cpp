/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

void test_mult_by_constant_scalar(alps::accumulators::accumulator_set & measurements, std::string name) {
	for (int i = 0; i < 1000; ++i)
		measurements[name] << 2. * (i % 2);
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);
	std::cout << results[name] * 2 << std::endl;
	std::cout << 2 * results[name] << std::endl;
	std::cout << results[name] / 2 << std::endl;
	std::cout << 2 / results[name] << std::endl;
}

void test_mult_by_constant_vector(alps::accumulators::accumulator_set & measurements, std::string name) {
	for (int i = 0; i < 1000; ++i)
		measurements[name] << std::vector<double>(10, 2. * (i % 2));
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);
	std::cout << results[name] * 2 << std::endl;
	std::cout << 2 * results[name] << std::endl;
	std::cout << results[name] / 2 << std::endl;
	std::cout << 2 / results[name] << std::endl;
}

#define ALPS_TEST_RUN_MUL_CONST_TEST(type, name)											\
	TEST(accumulators, mult_by_constant_scalar_ ## name) {									\
		alps::accumulators::accumulator_set measurements;									\
		measurements << alps::accumulators:: type <double>( #name "_scalar");				\
		test_mult_by_constant_scalar(measurements, #name "_scalar");						\
	}																						\
	TEST(accumulators, mult_by_constant_vector_ ## name) {									\
		alps::accumulators::accumulator_set measurements;									\
		measurements << alps::accumulators:: type <std::vector<double> >( #name "_vector");	\
		test_mult_by_constant_vector(measurements, #name "_vector");						\
	}

ALPS_TEST_RUN_MUL_CONST_TEST(MeanAccumulator, meanDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(NoBinningAccumulator, noBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(LogBinningAccumulator, logBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(FullBinningAccumulator, fullBinningDouble)
#undef ALPS_TEST_RUN_MUL_CONST_TEST

