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

        alps::params pars(argc,argv,"/parameters"); // reads from HDF5 if supplied
        std::string checkpoint_file = pars.get_base_name().substr(0, pars.get_base_name().find_last_of('.')) +  ".clone0.h5";

        alps::parameters_type<ising_sim>::type parameters(pars); // initializable from alps::params (and presumably is identical to it).
        ising_sim::define_parameters(parameters); // parameters are defined here inside, ahead of the constructor
        if (parameters.help_requested(std::cerr)) return 1; // Stop if help requested

        ising_sim sim(parameters); // some of the options are used in the constructor, so we needed to define them in advance

        if (parameters["continue"])
            sim.load(checkpoint_file);

        sim.run(alps::stop_callback(int(parameters["timelimit"])));

        sim.save(checkpoint_file);

        using alps::collect_results;
        alps::results_type<ising_sim>::type results = collect_results(sim);

        std::cout << results << std::endl;
        alps::hdf5::archive ar(parameters["output_file"], "w");
        ar["/parameters"] << parameters;
        ar["/simulation/results"] << results;

    } catch (std::exception const & e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
