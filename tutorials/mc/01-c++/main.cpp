/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <alps/mc/api.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/stop_callback.hpp>
#include "alps/utilities/remove_extensions.hpp"
// #include <alps/ngs/make_parameters_from_xml.hpp>

#include <boost/chrono.hpp>
// #include <boost/filesystem/path.hpp>

#include <string>
#include <iostream>
#include <stdexcept>

int main(int argc, const char *argv[]) {

     try {
        typedef alps::parameters_type<ising_sim>::type params_type;
        params_type parameters(argc, argv, "/parameters"); // reads from HDF5 if need be
        
        std::string checkpoint_file;
        if (parameters.is_restored()) {
            checkpoint_file = parameters.get_archive_name();
        } else {
            ising_sim::define_parameters(parameters);
            checkpoint_file=alps::remove_extensions(parameters.get_origin_name())+".clone0.h5";
        }
        if (parameters.help_requested(std::cerr)) return 1; // Stop if help requested.

        if (parameters["outputfile"].as<std::string>().empty()) {
            parameters["outputfile"] = alps::remove_extensions(parameters.get_origin_name()) + ".out.h5";
        }

        ising_sim sim(parameters);

        if (parameters.is_restored())
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
