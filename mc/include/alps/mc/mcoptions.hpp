/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#include <string>

namespace alps {

      class ALPS_DECL mcoptions {

        public:

            typedef enum { SINGLE, THREADED, MPI, HYBRID } execution_types;

            mcoptions(int argc, char* argv[]);

            bool valid;
            bool resume;
            std::size_t time_limit;
            std::string input_file;
            std::string output_file;
            std::string checkpoint_file;
            execution_types type;
    };
}
