#ifndef ALPS_CONV_PARAMS_INCLUDED
#define ALPS_CONV_PARAMS_INCLUDED

#include "alps/params.hpp"

namespace alps {
    params& define_convenience_parameters(params & parameters) {
        parameters
            // .define<std::string>("continue", "", "load simulation from the given checkpoint")
            .define<std::size_t>("timelimit", 0, "time limit for the simulation")
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
            return parameters;
    }
} // end of namespace alps
#endif
