/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* This is to test schedule checker */

#include <ctime>

#include <alps/mc/check_schedule.hpp>
#include <gtest/gtest.h>

class fake_timer {
  public:
    typedef std::time_t time_point_type;
    typedef double time_duration_type;

    static void reset(std::size_t ini) { now_=ini; }
    
    static void advance(std::size_t delta) { now_+=delta; }
    
    static time_point_type now_time() { return now_; }

    static time_duration_type time_diff(time_point_type t1, time_point_type t0)
    {
        return std::difftime(t1, t0);
    }
  private:
    static time_point_type now_;
};

fake_timer::time_point_type fake_timer::now_;

/// Test the `fake_timer` behavior
TEST(CheckScheduleTest, FakeTimerTest)
{
    const std::size_t step=10;
    
    fake_timer timer;
    timer.reset(1000);
    
    fake_timer timer2(timer);
    EXPECT_EQ(timer.now_time(), timer2.now_time());
    
    fake_timer::time_point_type t1=timer.now_time();
    timer.advance(step);
    fake_timer::time_point_type t2=timer.now_time();
    fake_timer::time_duration_type delta=timer.time_diff(t2, t1);

    EXPECT_EQ(std::difftime(t2, t1), delta);
    EXPECT_EQ(step, delta);

    EXPECT_EQ(timer.now_time(), timer2.now_time());
}

typedef alps::detail::generic_check_schedule<fake_timer> test_check_schedule;

TEST(CheckScheduleTest, UseFakeTimer)
{
    const std::size_t MIN_CHECK=3;
    const std::size_t MAX_CHECK=5;

    // First, our fake timer is at some zero time
    fake_timer timer;
    timer.reset(10000);
    
    test_check_schedule checker(MIN_CHECK, MAX_CHECK, timer);

    EXPECT_TRUE(checker.pending());

    checker.update(0.05);
    EXPECT_FALSE(checker.pending());

    timer.advance(MIN_CHECK+1);

    EXPECT_TRUE(checker.pending());
    EXPECT_TRUE(checker.pending());
    
    checker.update(0.1);
    EXPECT_FALSE(checker.pending());
    
    timer.advance(MIN_CHECK);
    EXPECT_FALSE(checker.pending());

    timer.advance(MAX_CHECK-MIN_CHECK+1);
    EXPECT_TRUE(checker.pending());

    checker.update(0.35);
    EXPECT_FALSE(checker.pending());
    
    timer.advance(6);
    EXPECT_TRUE(checker.pending());

    checker.update(0.9);
    EXPECT_FALSE(checker.pending());

    timer.advance(MIN_CHECK-1);
    EXPECT_FALSE(checker.pending());
    timer.advance(2);
    EXPECT_TRUE(checker.pending());
}

// WARNING: the following test relies on a real timing of the code,
// and thus:
// (a) Long (about 15 sec)
// (b) May be unreliable under high system load
// therefore it's disabled by default.
TEST(CheckScheduleTest, DISABLED_UseRealTimer)
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

/// Test that our POSIX timer wrapper works as expected
TEST(CheckSheduleTest, PosixTimerWrapper)
{
    std::time_t t0=std::time(0);
    EXPECT_NEAR(t0, alps::detail::posix_wall_clock::now_time(), 1) <<"Timer wrapper is not in sync with clock";

    std::time_t t1=t0;
    double delta=0;
    while (delta<2) {
        t1=std::time(0);
        delta=std::difftime(t1, t0);
    }

    EXPECT_NEAR(t1, alps::detail::posix_wall_clock::now_time(), 1) << "Timer wrapper counts time differently";
    EXPECT_EQ(delta, alps::detail::posix_wall_clock::time_diff(t1, t0)) << "Timer wrapper computes intervals differently";
}
