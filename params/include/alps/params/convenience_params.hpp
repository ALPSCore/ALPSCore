/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_CONV_PARAMS_INCLUDED
#define ALPS_CONV_PARAMS_INCLUDED

#include "alps/params.hpp"
#include "alps/utilities/fs/remove_extensions.hpp"
#include "alps/utilities/fs/get_basename.hpp"

namespace alps {
    /// @brief Defines a number of frequently-used parameters.
    /// Defines `size_t timelimt`, `string outputfile`, `string checkpoint`.
    inline params& define_convenience_parameters(params & parameters) {
        std::string origin;
        if (parameters.is_restored()) origin=parameters.get_archive_name();
        else if (parameters.get_ini_name_count()>0) origin=parameters.get_ini_name(0);
        else origin=parameters.get_argv0();
        
        const std::string basename=alps::fs::remove_extensions(alps::fs::get_basename(origin));
        parameters
            .define<std::size_t>("timelimit", 0, "time limit for the simulation")
            .define<std::string>("outputfile", basename+".out.h5", "name of the output file")
        ;
        const std::string base_outname=alps::fs::remove_extensions(alps::fs::get_basename(parameters["outputfile"]));
        parameters
            .define<std::string>("checkpoint", base_outname+".clone.h5", "name of the checkpoint file to save to")
        ;
        return parameters;
    }
} // end of namespace alps
#endif
