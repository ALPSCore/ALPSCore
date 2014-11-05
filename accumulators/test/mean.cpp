/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators/accumulator.hpp>
#include "gtest/gtest.h"


TEST(accumulator, mean_feature_scalar){

	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::MeanAccumulator<double>("obs1")
				 << alps::accumulators::MeanAccumulator<double>("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << 1.;
		EXPECT_EQ(measurements["obs1"].mean<double>() , 1.);
		measurements["obs2"] << i;
		EXPECT_EQ(measurements["obs2"].mean<double>() , double(i + 1) / 2.);
	}

	alps::accumulators::result_set results(measurements);
	EXPECT_EQ(results["obs1"].mean<double>() , 1.);
	EXPECT_EQ(results["obs2"].mean<double>() , 500.);
}

typedef long double longdouble;

void meas_test_body(alps::accumulators::accumulator_set &measurements){

        int L=10;

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << std::vector<double>(L, 1.);
		measurements["obs2"] << std::vector<double>(L, i);
                std::vector<double> mean_vec_1=measurements["obs1"].mean<std::vector<double> >();
                std::vector<double> mean_vec_2=measurements["obs2"].mean<std::vector<double> >();
                for(int j=0;j<mean_vec_1.size();++j){
		  EXPECT_EQ(mean_vec_1[j] , 1.);
		  EXPECT_EQ(mean_vec_2[j] , (i + 1) / 2.);
                }
	}

	alps::accumulators::result_set results(measurements);
        std::vector<double> mean_vec_1=results["obs1"].mean<std::vector<double> >();
        std::vector<double> mean_vec_2=results["obs2"].mean<std::vector<double> >();
        for(int i=0;i<mean_vec_1.size();++i){
	  EXPECT_EQ(mean_vec_1[i] , 1.);
          EXPECT_EQ(mean_vec_2[i] , 500.);
        }
}
#define ALPS_TEST_RUN_MEAN_TEST(A, T, N)														\
	TEST(accumulator, mean_feature_vector_ ## A ## N){											\
		alps::accumulators::accumulator_set measurements;										\
		measurements << alps::accumulators:: A < T >("obs1")									\
					 << alps::accumulators:: A < T >("obs2");									\
	  meas_test_body(measurements);																\
	}
#define ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(A)													\
	ALPS_TEST_RUN_MEAN_TEST(A, std::vector<float>, _f)											\
	ALPS_TEST_RUN_MEAN_TEST(A, std::vector<double>, _d)											\
	ALPS_TEST_RUN_MEAN_TEST(A, std::vector<longdouble>, _ld)

ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(MeanAccumulator)
ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(NoBinningAccumulator)
ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(LogBinningAccumulator)
ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(FullBinningAccumulator)

#undef ALPS_TEST_RUN_MEAN_TEST
#undef ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

