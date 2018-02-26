/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

#include <alps/testing/unique_file.hpp>

#include <boost/lambda/lambda.hpp>

#include "gtest/gtest.h"
// Simulation to measure e^(-x*x)
class my_sim_type : public alps::mcbase {

    public:

        my_sim_type(parameters_type const & params, std::size_t seed_offset = 42)
            : alps::mcbase(params, seed_offset)
            , count(0)
            , total_count(params["COUNT"])

        {
            measurements << alps::accumulators::FullBinningAccumulator<double>("SValue")
                         << alps::accumulators::FullBinningAccumulator<std::vector<double> >("VValue");
        }

        // if not compiled with mpi alps::mpi::communicator does not exists, 
        // so template the function
        template <typename Arg> my_sim_type(parameters_type const & params, Arg comm)
            : alps::mcbase(params, comm)
            , total_count(params["COUNT"])
        {
            measurements << alps::accumulators::FullBinningAccumulator<double>("SValue")
                         << alps::accumulators::FullBinningAccumulator<std::vector<double> >("VValue");
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

TEST(mc, sum_single){
    alps::parameters_type<my_sim_type>::type params;

    params["COUNT"]=1000;

    my_sim_type::define_parameters(params); // do parameters definitions
    my_sim_type my_sim(params); // create a simulation
    my_sim.run(alps::simple_time_callback(5)); // run the simulation for 1 second

    alps::results_type<my_sim_type>::type results = collect_results(my_sim); // collect the results

    std::cout << "e^(-x*x): " << results["SValue"] << std::endl;
    std::cout << "e^(-x*x): " << results["VValue"] << std::endl;
    alps::save_results(results, params, alps::testing::temporary_filename("sum_single.h5."), "/simulation/results");
}
