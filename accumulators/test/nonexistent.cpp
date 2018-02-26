/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include <iostream>
#include "gtest/gtest.h"

TEST(accumulator, nonexistent_acc){

	alps::accumulators::accumulator_set measurements;
	measurements << alps::accumulators::MeanAccumulator<double>("scalar");

        const alps::accumulators::accumulator_wrapper& null_acc=measurements["this_acc_doesnt_exist"];
        EXPECT_THROW(null_acc.result(), std::runtime_error);
        EXPECT_THROW(null_acc.reset(), std::runtime_error);
        EXPECT_THROW(null_acc.count(), std::runtime_error);
        EXPECT_THROW(null_acc.mean<double>(), std::runtime_error);
        EXPECT_THROW(null_acc.error<double>(), std::runtime_error);
        EXPECT_THROW(null_acc.print(std::cout), std::runtime_error);
}
