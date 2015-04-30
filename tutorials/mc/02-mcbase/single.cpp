/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <alps/mc/api.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/stop_callback.hpp>
// #include <alps/ngs/make_parameters_from_xml.hpp>

#include <boost/chrono.hpp>
#include <boost/filesystem/path.hpp>

#include <string>
#include <iostream>
#include <stdexcept>

int main(int argc, const char *argv[]) {

    try {
        alps::parameters_type<ising_sim>::type parameters(argc,argv,"/parameters"); // reads from HDF5 if supplied
        // if parameters are restored from the archive, all definitions are already there
        if (!parameters.is_restored()) {
          ising_sim::define_parameters(parameters); // parameters are defined here inside, ahead of the constructor
        }
        if (parameters.help_requested(std::cerr)) return EXIT_FAILURE; // Stop if help requested
        std::string checkpoint_file=parameters["checkpoint"];

        ising_sim sim(parameters); 

        if (parameters.is_restored()) // if the parameters are restored from a checkpoint file
            sim.load(checkpoint_file);

        sim.run(alps::stop_callback(int(parameters["timelimit"])));

        sim.save(checkpoint_file);

        using alps::collect_results;
        alps::results_type<ising_sim>::type results = collect_results(sim);

        std::cout << results << std::endl;
        alps::hdf5::archive ar(parameters["outputfile"], "w");
        ar["/parameters"] << parameters;
        ar["/simulation/results"] << results;

    } catch (std::exception const & e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
