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
TEST(accumulator, mean_feature_vector_MeanObserbale){
	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::MeanAccumulator<std::vector<double> >("obs1")
				 << alps::accumulators::MeanAccumulator<std::vector<double> >("obs2");
  meas_test_body(measurements);
}
TEST(accumulator, mean_feature_vector_NoBinningAccumulator){
	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("obs1")
				 << alps::accumulators::NoBinningAccumulator<std::vector<double> >("obs2");
  meas_test_body(measurements);
}
TEST(accumulator, mean_feature_vector_LogBinningAccumulator){
	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::LogBinningAccumulator<std::vector<double> >("obs1")
				 << alps::accumulators::LogBinningAccumulator<std::vector<double> >("obs2");
  meas_test_body(measurements);
}
TEST(accumulator, mean_feature_vector_FullBinningAccumulator){
	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::FullBinningAccumulator<std::vector<double> >("obs1")
				 << alps::accumulators::FullBinningAccumulator<std::vector<double> >("obs2");
  meas_test_body(measurements);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

