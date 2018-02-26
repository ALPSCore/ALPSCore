/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* This is to test timer/signal callback together with MC scheduler */

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/stop_callback.hpp>

#include <time.h>
#include <gtest/gtest.h>


class longish_sim : public alps::mcbase {
    int countdown_;
  public:
    static const int HOW_LONG=10;
    longish_sim(const parameters_type& p, std::size_t offset=0) : alps::mcbase(p,offset), countdown_(HOW_LONG)
    {}

    void update() { --countdown_; }

    void measure() {
        timespec requested;
        requested.tv_sec=1;
        requested.tv_nsec=0;
        nanosleep(&requested, 0);
    }

    double fraction_completed() const {
        return (countdown_<=0)?1.:0.;
    }

    unsigned long countdown() const { return countdown_; }
};

// parametrized by callback class
template <typename C>
class TestCallback : public ::testing::Test {
  public:
    typedef C callback_type;
    static const std::size_t TIMELIMIT=1;
    alps::params param_;
    longish_sim sim_;

    TestCallback() : param_(), sim_((param_["SEED"]=43,param_)) {}

    void timer_sim_test() {
        time_t start=time(0);
        sim_.run(callback_type(TIMELIMIT));
        time_t stop=time(0);
        double elapsed=difftime(stop, start);
        EXPECT_TRUE(sim_.countdown()>0) << "Simulation never timed out?";
        EXPECT_NEAR(TIMELIMIT, elapsed, 1.0);
    }
};

typedef ::testing::Types<alps::stop_callback, alps::simple_time_callback> CallbackTypes;
TYPED_TEST_CASE(TestCallback, CallbackTypes);

TYPED_TEST(TestCallback, TimerSimTest) { this->timer_sim_test(); }
