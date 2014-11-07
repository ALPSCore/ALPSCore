/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

// this must be first
#include <alps/utilities/boost_python.hpp>

#include <alps/utilities/import_numpy.hpp>

#include <boost/python/numeric.hpp>
#include <numpy/arrayobject.h>

namespace alps {
    namespace detail {

        ALPS_DECL void import_numpy() {
            static bool initialized = false;
            if (!initialized) {
                import_array();  
                boost::python::numeric::array::set_module_and_type("numpy", "ndarray");
                initialized = true;
            }
        }
    }
}
