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
#include <alps/mc/mpiadapter.hpp>

int main(int argc, char** argv)
{
    // Define the type for the simulation
    typedef alps::mcmpiadapter<ising_sim> my_sim_type;

    // Initialize the MPI environment, and obtain the WORLD communicator
    alps::mpi::environment env(argc, argv);
    alps::mpi::communicator comm;
    const int rank=comm.rank();
    const bool is_master=(rank==0);

    try {
        // Creates the parameters for the simulation
        // If an hdf5 file is supplied, reads the parameters there
        if (is_master) std::cout << "Initializing parameters..." << std::endl;

        // This constructor broadcasts to all processes
        alps::params parameters(argc, argv, comm);
        my_sim_type::define_parameters(parameters);

        if (parameters.help_requested(std::cout) ||
            parameters.has_missing(std::cout)) {
            return 1;
        }
    
        std::cout << "Creating simulation on rank " << rank << std::endl;
        my_sim_type sim(parameters, comm); 

        // If needed, restore the last checkpoint
        std::string checkpoint_file = parameters["checkpoint"].as<std::string>();
        if (!is_master) checkpoint_file+="."+boost::lexical_cast<std::string>(rank);
        
        if (parameters.is_restored()) {
            std::cout << "Restoring checkpoint from " << checkpoint_file
                      << " on rank " << rank << std::endl;
            sim.load(checkpoint_file);
        }

        // Run the simulation
        std::cout << "Running simulation on rank " << rank << std::endl;
        sim.run(alps::stop_callback(size_t(parameters["timelimit"])));

        // Checkpoint the simulation
        std::cout << "Checkpointing simulation to " << checkpoint_file
                  << " on rank " << rank << std::endl;
        sim.save(checkpoint_file);

        alps::results_type<my_sim_type>::type results = alps::collect_results(sim);

        // Print results
        if (is_master) {
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
        env.abort(2);
        return 2;
    } catch (...) {
        std::cout << "Unknown exception caught." << std::endl;
        env.abort(2);
        return 2;
    }
}
