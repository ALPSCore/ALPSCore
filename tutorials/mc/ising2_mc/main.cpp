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

int main(int argc, char** argv)
{
    // Define the type for the simulation
    typedef ising_sim my_sim_type;

    try {
    
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        std::cout << "Initializing parameters..." << std::endl;

        alps::params parameters(argc, argv);
        my_sim_type::define_parameters(parameters);

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }
    
        std::cout << "Creating simulation" << std::endl;
        my_sim_type sim(parameters); 

        // If needed, restore the last checkpoint
        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();
        
        if (parameters.is_restored()) {
            std::cout << "Restoring checkpoint from " << checkpoint_file
                      << std::endl;
            sim.load(checkpoint_file);
        }

        // Run the simulation
        std::cout << "Running simulation" << std::endl;
        sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

        // Checkpoint the simulation
        std::cout << "Checkpointing simulation to " << checkpoint_file
                  << std::endl;
        sim.save(checkpoint_file);

        alps::results_type<my_sim_type>::type results = alps::collect_results(sim);

        // Print results
        {
            using alps::accumulators::result_wrapper;
            std::cout << "All measured results:" << std::endl;
            std::cout << results << std::endl;
            
            std::cout << "Simulation ran for "
                      << results["Energy"].count()
                      << " steps." << std::endl;

            // Assign individual results to variables.
            const result_wrapper& mag4=results["Magnetization^4"];
            const result_wrapper& mag2=results["Magnetization^2"];

            // Derived result:
            const result_wrapper& binder_cumulant=1-mag4/(3*mag2*mag2);
            std::cout << "Binder cumulant: " << binder_cumulant
                      << " Relative error: "
                      << fabs(binder_cumulant.error<double>()/
                              binder_cumulant.mean<double>())
                      << std::endl;
            
            // Saving to the output file
            std::string output_file = parameters["outputfile"];
            alps::hdf5::archive ar(output_file, "w");
            ar["/parameters"] << parameters;
            ar["/simulation/results"] << results;
        }
        return 0;
    } catch (const std::runtime_error& exc) {
        std::cout << "Exception caught: " << exc.what() << std::endl;
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        return 2;
    }
}
