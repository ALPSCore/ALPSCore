/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* This is to test timer/signal callback */

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/stop_callback.hpp>

#include <time.h>
#include <gtest/gtest.h>

// parametrized by callback class
template <typename C>
class TestCallback : public ::testing::Test {
  public:
    typedef C callback_type;
    static const std::size_t TIMELIMIT=1;
    alps::params param_;

    TestCallback() {}

    void timer_test() {
        callback_type callback(TIMELIMIT);
        time_t start=time(0);
        double elapsed=0;
        while (!callback()) {
            time_t stop=time(0);
            elapsed=difftime(stop, start);
            ASSERT_GE(TIMELIMIT+1, elapsed) << "Failed to detect time out";
        }
        ASSERT_LE(TIMELIMIT-1, elapsed) << "Timed out too early";
    }
};

typedef ::testing::Types<alps::stop_callback, alps::simple_time_callback> CallbackTypes;
TYPED_TEST_CASE(TestCallback, CallbackTypes);

TYPED_TEST(TestCallback, TimerTest) { this->timer_test(); }
