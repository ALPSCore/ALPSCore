/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

TEST(accumulator, nonexistent_acc){

	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::MeanAccumulator<double>("scalar");

        EXPECT_ANY_THROW(measurements["this_acc_doesnt_exist"].mean<double>());
}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

