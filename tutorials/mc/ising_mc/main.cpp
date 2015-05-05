/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <iostream>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/stop_callback.hpp>

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
int main(int argc, const char* argv[])
{
    // Creates the parameters for the simulation
    // If an hdf5 file is supplied, reads the parameters there
    std::cout << "Initializing parameters..." << std::endl;
    alps::parameters_type<ising_sim>::type parameters(argc, argv, "/parameters");
    ising_sim::define_parameters(parameters);
    if (parameters.help_requested(std::cout)) {
        exit(0);
    }
    

    ising_sim sim(parameters); 

    // If needed, restore the last checkpoint
    std::string checkpoint_file = parameters["checkpoint"];
    if (parameters.is_restored()) {
        std::cout << "Restoring checkpoint from " << checkpoint_file << std::endl;
        sim.load(checkpoint_file);
    }

    // Run the simulation
    std::cout << "Running simulation..." << std::endl;
    sim.run(alps::stop_callback(int(parameters["timelimit"])));

    // Checkpoint the simulation
    std::cout << "Checkpointing simulation..." << std::endl;
    sim.save(checkpoint_file);

    alps::results_type<ising_sim>::type results = alps::collect_results(sim);

    // Print results
    std::cout << "Result:" << std::endl;
    std::cout << results << std::endl;

    // Saving to the output file
    alps::hdf5::archive ar(parameters["outputfile"], "w");
    ar["/parameters"] << parameters;
    ar["/simulation/results"] << results;

    return 0;
}
