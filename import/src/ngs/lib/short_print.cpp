/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/ngs/short_print.hpp>

namespace alps { 
    namespace detail {
        std::ostream & operator<<(std::ostream & os, short_print_proxy<float> const & v) {
            std::streamsize precision = os.precision(v.precision);
            os << v.value;
            os.precision(precision);
            return os;
        }
    
        std::ostream & operator<<(std::ostream & os, short_print_proxy<double> const & v) {
            std::streamsize precision = os.precision(v.precision);
            os << v.value;
            os.precision(precision);
            return os;
        }

        std::ostream & operator<<(std::ostream & os, short_print_proxy<long double> const & v) {
            std::streamsize precision = os.precision(v.precision);
            os << v.value;
            os.precision(precision);
            return os;
        }
    } 
}
