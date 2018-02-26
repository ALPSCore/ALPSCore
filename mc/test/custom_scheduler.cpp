/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/stop_callback.hpp>

#include <gtest/gtest.h>

class my_sim_type : public alps::mcbase {
    int _count;
  public:
    static const int MAXCOUNT=100;
    
    my_sim_type(const parameters_type& p, std::size_t offset=0) : alps::mcbase(p,offset), _count(0)
    {}

    void update() { ++_count; }

    void measure() {}

    double fraction_completed() const { return (_count<MAXCOUNT)?0:1; }

    int count() { return _count; }
};

class my_schecker_type {
  public:
    my_schecker_type() {}
    my_schecker_type(const alps::params&) {};
    bool pending() { return true; }
    void update(double /*f*/) {}
};

static bool stop_callback() { return false; }

TEST(CustomScheduler,Run) {
    typedef alps::mcmpiadapter<my_sim_type,my_schecker_type> sim_type;
    alps::mpi::communicator comm;
    alps::params p;
    sim_type::define_parameters(p);

    sim_type sim(p, comm, my_schecker_type());
    sim.run(stop_callback);

    EXPECT_EQ(sim_type::MAXCOUNT+0, sim.count());
}

TEST(CustomScheduler,Params) {
    typedef alps::mcmpiadapter<my_sim_type,my_schecker_type> sim_type;
    alps::mpi::communicator comm;
    alps::params p;
    sim_type::define_parameters(p);

    EXPECT_FALSE(p.defined("Tmin"));
    EXPECT_FALSE(p.defined("Tmax"));
}

int main(int argc, char**argv)
{
   alps::mpi::environment env(argc, argv, false);
   ::testing::InitGoogleTest(&argc, argv);
   return RUN_ALL_TESTS();
}    
