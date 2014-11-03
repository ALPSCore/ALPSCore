/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include "ising.hpp"

#include <alps/mc/api.hpp>
#include <alps/mc/parseargs.hpp>
#include <alps/mc/mpiadapter.hpp>
#include <alps/mc/stop_callback.hpp>
// #include <alps/ngs/make_parameters_from_xml.hpp>

#include <boost/chrono.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/path.hpp>

#include <string>
#include <iostream>
#include <stdexcept>

int main(int argc, char *argv[]) {

    try {
        boost::mpi::environment env(argc, argv);
        boost::mpi::communicator comm;

        alps::parseargs options(argc, argv);
        std::string checkpoint_file = options.input_file.substr(0, options.input_file.find_last_of('.')) 
                                    +  ".clone" + boost::lexical_cast<std::string>(comm.rank()) + ".h5";

        alps::parameters_type<ising_sim>::type parameters;
        // only load parameter if rank is 0 and brcast them to other ranks, els do nothing
        if (comm.rank() > 0)
            /* do nothing */ ;
        // else if (boost::filesystem::extension(options.input_file) == ".xml")
        //     parameters = alps::make_parameters_from_xml(options.input_file);
        else if (boost::filesystem::extension(options.input_file) == ".h5")
            alps::hdf5::archive(options.input_file)["/parameters"] >> parameters;
        else
            parameters = alps::parameters_type<ising_sim>::type(options.input_file);
        broadcast(comm, parameters);

        alps::mcmpiadapter<ising_sim> sim(parameters, comm, alps::check_schedule(options.tmin, options.tmax));

        if (options.resume)
            sim.load(checkpoint_file);

        // TODO: how do we handle signels in mpi context? do we want to handle these in the callback or in the simulation?
        // do not use stop_callback_mpi: we do not want an bcast after every sweep!
        //  Additionally this causes a race cond and deadlocks as mcmpiadapter::run will always call the stop_callback broadcast
        //  but only sometimes all_reduce on the fraction. Timers on different procs are not synchronized so they may not agree
        //  on the mpi call.
        sim.run(alps::stop_callback(comm, options.timelimit));

        sim.save(checkpoint_file);

        using alps::collect_results;
        alps::results_type<ising_sim>::type results = collect_results(sim);

        if (comm.rank() == 0) {
            std::cout << results << std::endl;
            alps::hdf5::archive ar(options.output_file, "w");
            ar["/parameters"] << parameters;
            ar["/simulation/results"] << results;
        }

    } catch (std::exception const & e) {
        std::cerr << "Caught exception: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
