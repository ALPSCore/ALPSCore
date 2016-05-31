/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <iostream>
#include <fstream>

#include <alps/accumulators.hpp>
#include <alps/mc/api.hpp>
#include <alps/mc/mcbase.hpp>
#include <alps/mc/stop_callback.hpp>

/**
 * This example shows how to setup a simple Monte Carlo simulation and retrieve some results.
 * <p>
 * The actual work is done inside the `ising_sim` class (see file "ising.cpp" and "ising.hpp"
 * in this directory). The simulation sets up a 2-dimensional Ising model and runs a Monte Carlo
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
int main(int argc, const char* argv[])
{
    typedef ising_sim my_sim_type;
    typedef std::vector<double> double_v;
    
    // Creates the parameters for the simulation
    // If an hdf5 file is supplied, reads the parameters from there
    std::cout << "Initializing parameters..." << std::endl;
    alps::parameters_type<my_sim_type>::type parameters(argc, argv, "/parameters");
    my_sim_type::define_parameters(parameters);
    if (parameters.help_requested(std::cout)) {
        exit(0);
    }

    std::cout << "Creating simulation..." << std::endl;
    my_sim_type sim(parameters); 

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

    alps::results_type<my_sim_type>::type results = alps::collect_results(sim);

    // Print results.
    std::cout << "All measured results:" << std::endl;
    std::cout << results << std::endl;
    // Access results individually: print correlation 
    std::ofstream corrf("correlations.dat");
    if (corrf) {
        corrf << "# Site CorrH CorrHErr CorrV CorrVErr CorrD CorrDErr\n";
        for (int i=0; i<parameters["length"]; ++i) {
            corrf << i << " "
                  << results["Correlations1"].mean<double_v>()[i] << " " << results["Correlations1"].error<double_v>()[i] << " "
                  << results["Correlations2"].mean<double_v>()[i] << " " << results["Correlations2"].error<double_v>()[i] << " "
                  << results["Correlations3"].mean<double_v>()[i] << " " << results["Correlations3"].error<double_v>()[i] << " "
                  << std::endl;
        }
    }
    corrf.close();
    
    std::cout << "Simulation ran for " << results["Energy"].count() << " steps.\n";
    const alps::accumulators::result_wrapper& mag4=results["Magnetization^4"];
    const alps::accumulators::result_wrapper& mag2=results["Magnetization^2"];

    // Derived result:
    const alps::accumulators::result_wrapper& binder_cumulant=1-mag4/(3*mag2*mag2);
    std::cout << "Binder cumulant: " << binder_cumulant
              << " Relative error: " << fabs(binder_cumulant.error<double>()/binder_cumulant.mean<double>())
              << std::endl;
    

    // Saving to the output file
    std::string output_file = parameters["outputfile"];
    alps::hdf5::archive ar(output_file, "w");
    ar["/parameters"] << parameters;
    ar["/simulation/results"] << results;

    return 0;
}
