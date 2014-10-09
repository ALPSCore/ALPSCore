/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/bitops.hpp>
#include "gtest/gtest.h"

TEST(bitops, main)
{
    uint32_t ui  = 5;
    uint64_t uli = 5;
    ui   <<= sizeof(uint32_t)*8-4;
    uli  <<= sizeof(uint64_t)*8-4;

    bool succ = alps::popcnt(ui)  == 2;
    succ     &= alps::popcnt(uli) == 2;
    ASSERT_EQ(succ, true);
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

