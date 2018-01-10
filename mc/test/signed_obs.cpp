/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
            measurements 
                << alps::accumulators::FullBinningAccumulator<double>("sign") 
                << alps::accumulators::FullBinningAccumulator<double>("x*sign");
        }
 
        void update() { count++; }
        void measure() { 
            measurements["sign"] << 1.0; 
            measurements["x*sign"] << 1.0;
        }
        double fraction_completed() const { return double (count/nsweeps); }
 
    private:
        int nsweeps;
        int count;
};
 
 
int main(int argc, char* argv[]) {
// TEST(mc, signed_obs) {
    alps::mpi::environment env(argc, argv);
    alps::mpi::communicator comm;
    alps::params p; 
    p["nsweeps"] = 100;
    sim1::define_parameters(p); // do parameters definitions
   alps::mcmpiadapter<sim1> sim (p,comm, alps::check_schedule(0.001, 60));
    sim.run(alps::stop_callback(1));
    if (comm.rank() == 0) {
        alps::results_type<sim1>::type res = alps::collect_results(sim);
        std::cout << res << std::endl;
    } else
        alps::collect_results(sim);
}
