/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* This is to test schedule checker */

// FIXME: this is provisional test; for real, we have to mock the timer
//        rather than rely on the timing of the test code!

#include <ctime>

#include <alps/mc/check_schedule.hpp>
#include <gtest/gtest.h>



TEST(CheckScheduleTest, CheckSchedule)
{
    const std::size_t MIN_CHECK=3;
    const std::size_t MAX_CHECK=5;

    alps::check_schedule checker(MIN_CHECK, MAX_CHECK);

    EXPECT_TRUE(checker.pending());

    std::time_t start=std::time(0);

    checker.update(0.05);
    while (!checker.pending()) {
        ASSERT_GT(MAX_CHECK+5, std::difftime(std::time(0), start)) << "Pending check never occured";
    }
    EXPECT_NEAR(MIN_CHECK, std::difftime(std::time(0), start), 1);

    start=std::time(0);
    checker.update(0.1);
    while (!checker.pending()) {
        ASSERT_GT(MAX_CHECK+5, std::difftime(std::time(0), start)) << "Pending check never occured";
    }
    EXPECT_NEAR(MAX_CHECK, std::difftime(std::time(0), start), 1);

    start=std::time(0);
    checker.update(0.35);
    while (!checker.pending()) {
        ASSERT_GT(MAX_CHECK+5, std::difftime(std::time(0), start)) << "Pending check never occured";
    }
    EXPECT_NEAR(5, std::difftime(std::time(0), start), 1);

    start=std::time(0);
    checker.update(0.9);
    while (!checker.pending()) {
        ASSERT_GT(MAX_CHECK+5, std::difftime(std::time(0), start)) << "Pending check never occured";
    }
    EXPECT_NEAR(MIN_CHECK, std::difftime(std::time(0), start), 1);
}
