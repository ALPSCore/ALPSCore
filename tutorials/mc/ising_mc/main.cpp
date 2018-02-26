/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <iostream>
#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

/**
 * This example shows how to setup a simple Monte Carlo simulation and retrieve some results.
 * <p>
 * The actual work is done inside the `ising_sim` class (see file "ising.cpp" and "ising.hpp"
 * in this directory). The simulation sets up a 1-dimensional Ising model and runs a Monte Carlo
 * simulation for the requested number of steps and at the requested temperature. This example
 * also shows how to read simulation parameters and save/restore the state of the simulation.
 * <p>
 * Run the example with `--help` argument to obtain the list of supported parameters.
 * <ul>
 *   <li>./ising_mc_--help</li>
 * </ul>
 * 
 * @param argc the number of arguments
 * @param argv the argument array
 * @return the exit code
 */
int main(int argc, char** argv)
{
    // Creates the parameters for the simulation
    // If an hdf5 file is supplied, reads the parameters there
    std::cout << "Initializing parameters..." << std::endl;
    alps::parameters_type<ising_sim>::type parameters(argc, argv, "/parameters");
    ising_sim::define_parameters(parameters);
    if (parameters.help_requested(std::cout)) {
        exit(0);
    }
    
    std::cout << "Creating simulation..." << std::endl;
    ising_sim sim(parameters); 

    // If needed, restore the last checkpoint
    std::string checkpoint_file = parameters["checkpoint"];
    if (parameters.is_restored()) {
        std::cout << "Restoring checkpoint from " << checkpoint_file << std::endl;
        sim.load(checkpoint_file);
    }

    // Run the simulation
    std::cout << "Running simulation..." << std::endl;
    sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

    // Checkpoint the simulation
    std::cout << "Checkpointing simulation..." << std::endl;
    sim.save(checkpoint_file);

    alps::results_type<ising_sim>::type results = alps::collect_results(sim);

    // Print results
    std::cout << "Result:" << std::endl;
    std::cout << results << std::endl;

    // Saving to the output file
    std::string output_file = parameters["outputfile"];
    alps::hdf5::archive ar(output_file, "w");
    ar["/parameters"] << parameters;
    ar["/simulation/results"] << results;

    return 0;
}
