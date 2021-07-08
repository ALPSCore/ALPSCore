/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/config.hpp>
#include <alps/accumulators.hpp>
#include "gtest/gtest.h"

namespace aa=alps::accumulators;

TEST(accumulators, floatVectorsNoCompile) {
#ifdef ALPS_TEST_EXPECT_COMPILE_FAILURE
    typedef std::vector<float> float_v;
    auto named_acc=aa::NoBinningAccumulator<float_v>("float_vector");
    FAIL() << "This test should have never compiled";
#else
    SUCCEED();
#endif
}
