/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

void test_divide_accumulators_scalar(alps::accumulators::accumulator_set & measurements, std::string name) {
	std::string name1 = name + "_1_scalar";
	std::string name2 = name + "_2_scalar";
	for (int i = 0; i < 1000; ++i) {
		measurements[name1] << (i % 2) + 1;
		measurements[name2] << 2. * (i % 2);
	}
        // FIXME: test rather than print!
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);
	std::cout << results[name1] * results[name2] << std::endl;
	std::cout << results[name1] / results[name2] << std::endl;

    double div_mean = (results[name1] / results[name2]).mean<double>();
    double div_mean2 = results[name1].mean<double>() / results[name2].mean<double>();
    std::cout << div_mean << " == " << div_mean2 << std::endl;

    ASSERT_EQ(boost::math::isnan(div_mean), false);
    ASSERT_EQ(boost::math::isinf(div_mean), false);
    ASSERT_NEAR(div_mean, div_mean2, 5e-4);
}

void test_divide_accumulators_vector(alps::accumulators::accumulator_set & measurements, std::string name) {
	std::string name1 = name + "_1_vector";
	std::string name2 = name + "_2_vector";
        unsigned int L=10;
	for (unsigned int i = 0; i < 1000; ++i) {
		measurements[name1] << std::vector<double>(L, (i % 2) + 1);
		measurements[name2] << std::vector<double>(L, 2. * (i % 2));
	}
	std::cout << measurements << std::endl;
	alps::accumulators::result_set results(measurements);
	std::cout << results[name1] * results[name2] << std::endl;
	std::cout << results[name1] / results[name2] << std::endl;
        std::vector<double> mean_vec=(results[name1] / results[name2]).mean<std::vector<double> >();
        ASSERT_EQ(mean_vec.size(), L);
        for(unsigned int i=0;i<L;++i) {
          EXPECT_EQ(mean_vec[i], mean_vec[0]);
        }
        alps::accumulators::result_wrapper r=results[name1];
        EXPECT_EQ(r.mean<std::vector<double> >().size(), L);
        r/=results[name2];
        EXPECT_EQ(r.mean<std::vector<double> >().size(), L);
}

#define ALPS_TEST_RUN_MUL_CONST_TEST(type, name)												\
	TEST(accumulators, divide_accumulators_scalar ## name) {										\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: type <double>( #name "_1_scalar") 					\
					 << alps::accumulators:: type <double>( #name "_2_scalar");					\
		test_divide_accumulators_scalar(measurements, #name);										\
	}																							\
	TEST(accumulators, divide_accumulators_vector ## name) {										\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: type <std::vector<double> >( #name "_1_vector")	\
					 << alps::accumulators:: type <std::vector<double> >( #name "_2_vector");	\
		test_divide_accumulators_vector(measurements, #name);										\
	}

ALPS_TEST_RUN_MUL_CONST_TEST(MeanAccumulator, meanDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(NoBinningAccumulator, noBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(LogBinningAccumulator, logBinningDouble)
ALPS_TEST_RUN_MUL_CONST_TEST(FullBinningAccumulator, fullBinningDouble)
#undef ALPS_TEST_RUN_MUL_CONST_TEST
