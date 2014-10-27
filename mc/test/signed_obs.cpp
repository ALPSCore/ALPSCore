/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/mcbase.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/api.hpp>
#include <alps/accumulators.hpp>
#include <alps/params.hpp>
#include <alps/mc/stop_callback.hpp>
 
class sim1 : public alps::mcbase {
    public:
        
        sim1(parameters_type const & p, std::size_t seed_offset = 0):
            alps::mcbase(p, seed_offset),
            nsweeps(p["nsweeps"]), 
            count(0) 
        {
            measurements << 
                alps::accumulators::RealObservable("e1") <<
                alps::accumulators::SignedRealObservable("e2");
        }
 
        void update() { count++; }
        void measure() { 
            measurements["e1"] << 1.0; 
            measurements["e2"](1.0, 1); // measure value and weight
        }
        double fraction_completed() const { return double (count/nsweeps); }
 
    private:
        int nsweeps;
        int count;
};
 
 
int main(int argc, char* argv[]) {
// TEST(mc, signed_obs) {
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator comm;
    alps::params p; 
    p["nsweeps"] = 100;
    alps::mcmpiadapter<sim1> sim (p,comm, alps::check_schedule(0.001, 60));
    sim.run(alps::stop_callback(1));
    if (comm.rank() == 0) {
        alps::results_type<sim1>::type res = alps::collect_results(sim);
        std::cout << res << std::endl;
    } else
        alps::collect_results(sim);
}