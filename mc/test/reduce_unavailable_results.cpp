/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/stop_callback.hpp>

#include "alps/utilities/mpi.hpp"

#include <alps/utilities/boost_mpi.hpp>
#include <alps/accumulators.hpp>
#include <alps/params.hpp>

#include "alps/utilities/gtest_par_xml_output.hpp"
#include "gtest/gtest.h"
//#include "mpi_guard.hpp"

class sim1 : public alps::mcbase {
    public:
        
        sim1(parameters_type const & p, std::size_t seed_offset = 0):
            alps::mcbase(p, seed_offset),
            nsweeps(p["nsweeps"]), 
            count(0),
            measuring(p["measuring"]),
            comm()
        {
            measurements << alps::accumulators::NoBinningAccumulator<std::vector<double> >("e1");
        }
 
        void update() {
            count++;
        }

        void measure() {
            if (measuring) {
              measurements["e1"] << std::vector<double>(5, 1.0);
            }
        }

        double fraction_completed() const { return (1.*count)/nsweeps; }
 
    private:
        int nsweeps;
        int count;
        bool measuring;
        alps::mpi::communicator comm;
};

TEST(mc, reduce_unavailable_results_test){
        alps::mpi::communicator c;
        alps::mcbase::parameters_type p;

        p["nsweeps"] = 100;
        sim1::define_parameters(p); // do parameters definitions

        if (c.size() > 1) {
            //measurement only on master process (it should throw)
            {
                p["measuring"] = (c.rank() == 0);
                alps::mcmpiadapter<sim1> sim(p, c, alps::check_schedule(0.001, 60));
                sim.run(alps::stop_callback(1));
                c.barrier();
                ASSERT_ANY_THROW({
                                     if (c.rank() == 0) {
                                         alps::results_type<sim1>::type res = alps::collect_results(sim);
                                     } else {
                                         alps::collect_results(sim);
                                     }
                                 });
            }


            //not measured on any process (it should NOT throw)
            {
                p["measuring"] = false;
                alps::mcmpiadapter<sim1> sim(p, c, alps::check_schedule(0.001, 60));
                sim.run(alps::stop_callback(1));
                c.barrier();
                ASSERT_NO_THROW({
                                    if (c.rank() == 0) {
                                        alps::results_type<sim1>::type res = alps::collect_results(sim);
                                    } else {
                                        alps::collect_results(sim);
                                    }
                                });
            }


            //measured on all processes (it should NOT throw)
            {
                p["measuring"] = true;
                alps::mcmpiadapter<sim1> sim(p, c, alps::check_schedule(0.001, 60));
                sim.run(alps::stop_callback(1));
                c.barrier();
                ASSERT_NO_THROW({
                                    if (c.rank() == 0) {
                                        alps::results_type<sim1>::type res = alps::collect_results(sim);
                                    } else {
                                        alps::collect_results(sim);
                                    }
                                });
            }

        }
}

int main(int argc, char* argv[]) {
    alps::mpi::environment env(argc, argv);
    alps::gtest_par_xml_output tweak;
    tweak(alps::mpi::communicator().rank(), argc, argv);
    ::testing::InitGoogleTest(&argc, argv);

    const int rc = RUN_ALL_TESTS();

    return rc;
}
