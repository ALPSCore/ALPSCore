/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators_.hpp>
#include "gtest/gtest.h"

typedef long double longdouble;
double prec=1e-12;

template<typename A, typename T> void mean_test_body_scalar() {

	alps::accumulator_set measurements;
	measurements << A("obs1") << A("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << T(1);
		EXPECT_NEAR(measurements["obs1"].mean<double>() , T(1) , prec);
		measurements["obs2"] << T(i);
		EXPECT_NEAR(measurements["obs2"].mean<double>() , T(i + 1) / 2 , prec);
	}

	alps::result_set results(measurements);
	EXPECT_NEAR(results["obs1"].mean<double>() , T(1) , prec);
	EXPECT_NEAR(results["obs2"].mean<double>() , T(500) , prec);
}

template<typename A, typename T> void mean_test_body_vector() {

	alps::accumulator_set measurements;
	measurements << A("obs1") << A("obs2");

	unsigned int L = 10;

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << std::vector<T>(L, T(1.));
		measurements["obs2"] << std::vector<T>(L, T(i));
		std::vector<T> mean_vec_1=measurements["obs1"].mean<std::vector<T> >();
		std::vector<T> mean_vec_2=measurements["obs2"].mean<std::vector<T> >();
		for(size_t j=0;j<mean_vec_1.size();++j){
			EXPECT_NEAR(mean_vec_1[j] , T(1.) , prec);
			EXPECT_NEAR(mean_vec_2[j] , T(i + 1) / 2 , prec);
		}
	}

	alps::result_set results(measurements);
		std::vector<T> mean_vec_1=results["obs1"].mean<std::vector<T> >();
		std::vector<T> mean_vec_2=results["obs2"].mean<std::vector<T> >();
                EXPECT_EQ(mean_vec_1.size(), L);
                EXPECT_EQ(mean_vec_2.size(), L);
		for(size_t i=0;i<mean_vec_1.size();++i){
	  		EXPECT_NEAR(mean_vec_1[i] , T(1.) , prec);
			EXPECT_NEAR(mean_vec_2[i] , T(500.) , prec);
	}
}

#define ALPS_TEST_RUN_MEAN_TEST(A, T, N)														\
	TEST(accumulator, mean_feature_scalar_ ## A ## _ ## N){										\
	  	mean_test_body_scalar<alps:: A < T >, T >();											\
	}																							\
	TEST(accumulator, mean_feature_vector_ ## A ## _vector_ ## N){								\
	  	mean_test_body_vector<alps:: A <std::vector< T > >, T >();								\
	}

#define ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(A)													\
	ALPS_TEST_RUN_MEAN_TEST(A, double, double)													\
	ALPS_TEST_RUN_MEAN_TEST(A, longdouble, long_double)
//	ALPS_TEST_RUN_MEAN_TEST(A, float, float)

ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(MeanAccumulator)
ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(NoBinningAccumulator)
ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(LogBinningAccumulator)
ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE(FullBinningAccumulator)

#undef ALPS_TEST_RUN_MEAN_TEST
#undef ALPS_TEST_RUN_MEAN_TEST_EACH_TYPE
