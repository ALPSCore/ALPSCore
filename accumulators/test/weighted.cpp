/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators/accumulator.hpp>
#include "gtest/gtest.h"

TEST(accumulators, SignedSimpleRealObservable){
	alps::accumulator::accumulator_set measurements;
	measurements << alps::accumulator::SignedSimpleRealObservable("obs1")
				<< alps::accumulator::SignedSimpleRealObservable("obs2")
				<< alps::accumulator::SignedSimpleRealObservable("obs3")
				<< alps::accumulator::SignedSimpleRealObservable("obs4");

	for (int i = 1; i < 1000; ++i) {
		measurements["obs1"](1., 1.);
		measurements["obs2"](1., i % 2 ? 1. : -1);
		measurements["obs3"](i, 1.);
		measurements["obs4"](i, i % 2 ? 1. : -1);
	}
	EXPECT_EQ(measurements["obs1"].mean<double>() , 1.);
	EXPECT_EQ(measurements["obs2"].mean<double>() , 1.);
	EXPECT_EQ(measurements["obs3"].mean<double>() , 500.);
	EXPECT_EQ(measurements["obs4"].mean<double>() , 500.);

	alps::accumulator::result_set results(measurements);
	EXPECT_EQ(results["obs1"].mean<double>() , 1.);
	EXPECT_EQ(results["obs2"].mean<double>() , 1.);
	EXPECT_EQ(results["obs3"].mean<double>() , 500.);
	EXPECT_EQ(results["obs4"].mean<double>() , 500.);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
