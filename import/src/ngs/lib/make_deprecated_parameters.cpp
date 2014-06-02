/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/ngs/make_deprecated_parameters.hpp>

#include <string>
#include <sstream>

namespace alps {

    Parameters make_deprecated_parameters(params const & arg) {
        Parameters par;
        for (params::const_iterator it = arg.begin(); it != arg.end(); ++it){
            std::stringstream s;
            s<<it->second;
            par.push_back(it->first,s.str());
        }
        return par;
    }
}
