/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_NUMERIC_MULTIARRAY_HEADER
#define ALPS_NGS_NUMERIC_MULTIARRAY_HEADER

#include <alps/multi_array/functions.hpp>

// Import multi_array functions into ngs::numeric namespace.
namespace alps {
    namespace ngs {
        namespace numeric {
            
            using alps::sin;
            using alps::cos;
            using alps::tan;
            using alps::sinh;
            using alps::cosh;
            using alps::tanh;
            using alps::asin;
            using alps::acos;
            using alps::atan;
            using alps::abs;
            using alps::sqrt;
            using alps::exp;
            using alps::log;
            using alps::fabs;

            using alps::sq;
            using alps::cb;
            using alps::cbrt;
            
            using alps::pow;
            using alps::sum;
        }
    }
}

#endif
