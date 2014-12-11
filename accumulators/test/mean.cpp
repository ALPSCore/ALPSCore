/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

typedef long double longdouble;

template<typename A, typename T> void mean_test_body_scalar() {

	alps::accumulators::accumulator_set measurements;
	measurements << A("obs1") << A("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << T(1);
		EXPECT_EQ(measurements["obs1"].mean<double>() , T(1));
		measurements["obs2"] << T(i);
		EXPECT_EQ(measurements["obs2"].mean<double>() , T(i + 1) / 2);
	}

	alps::accumulators::result_set results(measurements);
	EXPECT_EQ(results["obs1"].mean<double>() , T(1));
	EXPECT_EQ(results["obs2"].mean<double>() , T(500));
}

template<typename A, typename T> void mean_test_body_vector() {

	alps::accumulators::accumulator_set measurements;
	measurements << A("obs1") << A("obs2");

	int L = 10;

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << std::vector<T>(L, T(1.));
		measurements["obs2"] << std::vector<T>(L, T(i));
		std::vector<T> mean_vec_1=measurements["obs1"].mean<std::vector<T> >();
		std::vector<T> mean_vec_2=measurements["obs2"].mean<std::vector<T> >();
		for(int j=0;j<mean_vec_1.size();++j){
			EXPECT_EQ(mean_vec_1[j] , T(1.));
			EXPECT_EQ(mean_vec_2[j] , T(i + 1) / 2);
		}
	}

	alps::accumulators::result_set results(measurements);
		std::vector<T> mean_vec_1=results["obs1"].mean<std::vector<T> >();
		std::vector<T> mean_vec_2=results["obs2"].mean<std::vector<T> >();
		for(int i=0;i<mean_vec_1.size();++i){
	  		EXPECT_EQ(mean_vec_1[i] , T(1.));
			EXPECT_EQ(mean_vec_2[i] , T(500.));
	}
}

#define ALPS_TEST_RUN_MEAN_TEST(A, T, N)														\
	TEST(accumulator, mean_feature_scalar_ ## A ## _ ## N){										\
	  	mean_test_body_scalar<alps::accumulators:: A < T >, T >();								\
	}																							\
	TEST(accumulator, mean_feature_vector_ ## A ## _vector_ ## N){								\
	  	mean_test_body_vector<alps::accumulators:: A <std::vector< T > >, T >();				\
	}

#define ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(A)													\
	ALPS_TEST_RUN_MEAN_TEST(A, float, float)													\
	ALPS_TEST_RUN_MEAN_TEST(A, double, double)													\
	ALPS_TEST_RUN_MEAN_TEST(A, longdouble, long_double)

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

