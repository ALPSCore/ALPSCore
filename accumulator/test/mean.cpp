/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.h>
#include <alps/accumulator/accumulator.hpp>
#include "gtest/gtest.h"


TEST(accumulator, mean_feature_scalar){

	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealObservable("obs1")
				 << alps::accumulator::RealObservable("obs2");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"] << 1.;
		EXPECT_EQ(measurements["obs1"].mean<double>() , 1.);
		measurements["obs2"] << i;
		EXPECT_EQ(measurements["obs2"].mean<double>() , double(i + 1) / 2.);
	}

	alps::accumulator::result_set results(measurements);
	EXPECT_EQ(results["obs1"].mean<double>() , 1.);
	EXPECT_EQ(results["obs2"].mean<double>() , 500.);
}

void meas_test_body(alps::accumulator::accumulator_set &measurements){

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

	alps::accumulator::result_set results(measurements);
        std::vector<double> mean_vec_1=results["obs1"].mean<std::vector<double> >();
        std::vector<double> mean_vec_2=results["obs2"].mean<std::vector<double> >();
        for(int i=0;i<mean_vec_1.size();++i){
	  EXPECT_EQ(mean_vec_1[i] , 1.);
          EXPECT_EQ(mean_vec_2[i] , 500.);
        }
}
TEST(accumulator, mean_feature_vector_RealVectorObservable){
	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::RealVectorObservable("obs1")
				 << alps::accumulator::RealVectorObservable("obs2");
  meas_test_body(measurements);
}
TEST(accumulator, mean_feature_vector_SimpleRealVectorObservable){
	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::SimpleRealVectorObservable("obs1")
				 << alps::accumulator::SimpleRealVectorObservable("obs2");
  meas_test_body(measurements);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

