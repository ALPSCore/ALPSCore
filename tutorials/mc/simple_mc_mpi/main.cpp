/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>
#include <alps/mc/mpiadapter.hpp>

// Simulation class
// We extend alps::mcbase, which is the base class of all Monte Carlo simulations.
// Typically one would put this in a separate cpp file. Since this is a basic
// example, we include everything needed in a single file to help beginners to
// get oriented.
class my_sim_type : public alps::mcbase {
    
    // The internal state of our simulation
    private:
        // The current MC step
        int count;
        // The total MC step to do
        int total_count;
        // The value calculated for the current MC step
        double value;

    public:

        // The constructor for our simulation
        // We always need the parameters and the seed as we need to pass it to
        // the alps::mcbase constructor. We also initialize count to 0
        // and total_count based on the value of the nSteps parameter
        my_sim_type(parameters_type const & params, std::size_t seed_offset = 42)
            : alps::mcbase(params, seed_offset)
            , total_count(params["nSteps"])
            , count(0)
        {
            measurements << alps::accumulators::FullBinningAccumulator<double>("X")
                         << alps::accumulators::FullBinningAccumulator<double>("X2");
        }

        // This performs the actual calculation at each MC step.
        // In this example we simply take a value from a uniform distribution.
        void update() {
            value = random();
        };

        // This collects the measurements at each MC step.
        void measure() {
            // Increase the count
            count++;
            // Collect the value and the value squared
            measurements["X"] << value;
            measurements["X2"] << value*value;
        };

        // This must return a number from 0.0 to 1.0 that says how much
        // of the simulation has been completed
        double fraction_completed() const {
            return count / double(total_count);
        }

};


/**
 * This example shows how to setup a simple simulation and retrieve some results.
 * The simulation just takes samples from a uniform distribution from 0.0 to 
 * 1.0, and estimates the average and the variance. Full binning
 * accumulators are used to collect X and X^2 at each MC step, and to
 * calculate the variance at the end of the simulation.
 * <p>
 * Run the example with different arguments combinations. For example:
 * <ul>
 *   <li>./simple_mc_--help</li>
 *   <li>./simple_mc </li>
 *   <li>./simple_mc --nSteps 10000 </li>
 * </ul>
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, char** argv)
{
    // Use the MPI adapter class instead of the original class:
    typedef alps::mcmpiadapter<my_sim_type> my_sim_type;

    // Initialize the MPI environment:
    alps::mpi::environment env(argc,argv);
    // Obtain the MPI communicator (MPI_COMM_WORLD by default):
    alps::mpi::communicator comm;
    
    // Creates the parameters for the simulation
    std::cout << "Initializing parameters..." << std::endl;
    alps::parameters_type<my_sim_type>::type params(argc, argv, comm);

    // Define the parameters for our simulation, including the ones for the
    // base class
    params.define("nSteps", 1000, "Number of MC steps to perform");
    my_sim_type::define_parameters(params);
    if (params.help_requested(std::cout)) {
        return 0;
    }
    
    // Create and run the simulation
    std::cout << "Running simulation on rank " << comm.rank() << std::endl;
    my_sim_type my_sim(params,comm);
    my_sim.run(alps::stop_callback(5));

    // Collect the results from the simulation
    std::cout << "Rank " << comm.rank() << " has finished. Collecting results..." << std::endl;
    alps::results_type<my_sim_type>::type results = alps::collect_results(my_sim);

    // Print the mean and the standard deviation.
    // Only master has all the results!
    if (comm.rank()==0) {
        std::cout << "Results:" << std::endl;
        std::cout << "The simulation ran for " << results["X"].count() << " steps." << std::endl;
        std::cout << " mean: " << results["X"] << std::endl;
        std::cout << " variance: " << results["X2"] - results["X"]*results["X"] << std::endl;
    }
    return 0;
}
