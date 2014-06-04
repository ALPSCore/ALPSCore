/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/ngs/config.hpp>
#include <string>

namespace alps {

	struct ALPS_DECL parseargs {
	    parseargs(int argc, char *argv[]);

	    bool resume;

	    std::size_t timelimit;
	    std::size_t tmin;
	    std::size_t tmax;

	    std::string input_file;
	    std::string output_file;
	};
}
