/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utilities/signal.hpp>
#include <alps/utilities/remove_extensions.hpp>
#include <alps/mc/mcbase.hpp>
// #include "boost/filesystem/path.hpp"

namespace alps {

    mcbase::mcbase(parameters_type const & parms, std::size_t seed_offset)
        : parameters(parms)
          // , params(parameters) // TODO: remove, deprecated!
        , random(std::size_t(parameters["SEED"]) + seed_offset)
    {
        alps::signal::listen();
    }

    void mcbase::define_parameters(parameters_type & parameters) {
        parameters
            // .define<std::string>("continue", "", "load simulation from the given checkpoint")
            .define<std::size_t>("timelimit", 0, "time limit for the simulation")
            .define<std::size_t>("Tmin", 1, "minimum time to check if simulation has finished")
            .define<std::size_t>("Tmax", 600, "maximum time to check if simulation has finished")
            .define<std::size_t>("SEED", 42, "PRNG seed")
            .define<std::string>("outputfile", "*.out.h5", "name of the output file")
            .define<std::string>("checkpoint", "*.clone.h5", "name of the checkpoint file to save to")
        ;
        // FIXME: this is a hack. I need a method to see if a parameter is actually supplied.
        if (parameters["outputfile"].as<std::string>()[0]=='*') {
            parameters["outputfile"]=alps::remove_extensions(parameters.get_origin_name())+".out.h5";
        }
        // FIXME: this is a hack. I need a method to see if a parameter is actually supplied.
        if (parameters["checkpoint"].as<std::string>()[0]=='*') {
            parameters["checkpoint"]=alps::remove_extensions(parameters.get_origin_name())+".clone.h5";
        }
    }

    void mcbase::save(boost::filesystem::path const & filename) const {
        alps::hdf5::archive ar(filename, "w");
        ar["/simulation/realizations/0/clones/0"] << *this;
    }

    void mcbase::load(boost::filesystem::path const & filename) {
        alps::hdf5::archive ar(filename);
        ar["/simulation/realizations/0/clones/0"] >> *this;
    }

    bool mcbase::run(boost::function<bool ()> const & stop_callback) {
        bool stopped = false;
        while(!(stopped = stop_callback()) && fraction_completed() < 1.) {
            update();
            measure();
        }
        return !stopped;
    }

    // implement a nice keys(m) function
    mcbase::result_names_type mcbase::result_names() const {
        result_names_type names;
        for(observable_collection_type::const_iterator it = measurements.begin(); it != measurements.end(); ++it)
            names.push_back(it->first);
        return names;
    }

    mcbase::result_names_type mcbase::unsaved_result_names() const {
        return result_names_type(); 
    }

    mcbase::results_type mcbase::collect_results() const {
        return collect_results(result_names());
    }

    mcbase::results_type mcbase::collect_results(result_names_type const & names) const {
        results_type partial_results;
        for(result_names_type::const_iterator it = names.begin(); it != names.end(); ++it){
                partial_results.insert(*it, measurements[*it].result());
        }
        return partial_results;
    }

    void mcbase::save(alps::hdf5::archive & ar) const {
        ar["/parameters"] << parameters;
        ar["measurements"] << measurements;
        ar["checkpoint"] << random;
    }

    void mcbase::load(alps::hdf5::archive & ar) {
        ar["/parameters"] >> parameters;
        ar["measurements"] >> measurements;
        ar["checkpoint"] >> random;
    }

}
