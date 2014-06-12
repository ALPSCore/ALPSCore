/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/mcbase.hpp>
#include <alps/mc/api.hpp>
#include <alps/utility/parseargs.hpp>
#include <alps/utility/temporary_filename.hpp>
#include <alps/mc/mcmpiadapter.hpp>
#include <alps/utility/stop_callback.hpp>

#include <boost/lambda/lambda.hpp>

#include "gtest/gtest.h"
// Simulation to measure e^(-x*x)
class my_sim_type : public alps::mcbase {

    public:

        my_sim_type(parameters_type const & params, std::size_t seed_offset = 42)
            : alps::mcbase(params, seed_offset)
            , total_count(params["COUNT"])

        {
            measurements << alps::accumulator::RealObservable("SValue")
                         << alps::accumulator::RealVectorObservable("VValue");
        }

        // if not compiled with mpi boost::mpi::communicator does not exists, 
        // so template the function
        template <typename Arg> my_sim_type(parameters_type const & params, Arg comm)
            : alps::mcbase(params, comm)
            , total_count(params["COUNT"])
        {
            measurements << alps::accumulator::RealObservable("SValue")
                         << alps::accumulator::RealVectorObservable("VValue");
        }

        // do the calculation in this function
        void update() {
            double x = random();
            value = exp(-x * x);
        };

        // do the measurements here
        void measure() {
            ++count;
            measurements["SValue"] << value;
            measurements["VValue"] << std::vector<double>(3, value);
        };

        double fraction_completed() const {
            return count / double(total_count);
        }

    private:
        int count;
        int total_count;
        double value;
};

TEST(mc, sum_mpi){
        boost::mpi::environment env;
        boost::mpi::communicator c;

        alps::mcbase::parameters_type params;
        params["COUNT"]=1000;
        broadcast(c, params);
        
        int t_min_check=1, t_max_check=1, timelimit=1;

        alps::mcmpiadapter<my_sim_type> my_sim(params, c, alps::check_schedule(t_min_check, t_max_check)); // creat a simulation

        my_sim.run(alps::stop_callback(c, timelimit)); // run the simulation

        using alps::collect_results;

        if (c.rank() == 0) { // print the results and save it to hdf5
            alps::results_type<alps::mcmpiadapter<my_sim_type> >::type results = collect_results(my_sim);
            std::cout << "e^(-x*x): " << results["SValue"] << std::endl;
            std::cout << "e^(-x*x): " << results["VValue"] << std::endl;
            using std::sin;
            std::cout << results["SValue"] + 1 << std::endl;
            std::cout << results["SValue"] + results["SValue"] << std::endl;
            save_results(results, params, alps::temporary_filename("sum_mpi") , "/simulation/results");
        } else
            collect_results(my_sim);
}
