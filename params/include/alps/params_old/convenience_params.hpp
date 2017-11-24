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
        std::string basename=alps::fs::remove_extensions(alps::fs::get_basename(parameters.get_origin_name()));
        parameters
            .define<std::size_t>("timelimit", 0, "time limit for the simulation")
            .define<std::string>("outputfile", basename+".out.h5", "name of the output file")
            .define<std::string>("checkpoint", basename+".clone.h5", "name of the checkpoint file to save to")
        ;
        // FIXME: we might want to base checkpoint name on the output file name rather than origin name
        //        (especially if there is no origin name: e.g. everything is passed via the command line
        //         and argv[0] is wrong/unavailable)
        return parameters;
    }
} // end of namespace alps
#endif
