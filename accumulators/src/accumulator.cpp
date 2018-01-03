/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/accumulators.hpp>
#include <sstream>

namespace alps {
    namespace accumulators {

#define EXTERNAL_FUNCTION(FUN)                                  \
        result_wrapper FUN (result_wrapper const & arg) {       \
            return arg. FUN ();                                 \
        }
        EXTERNAL_FUNCTION(sin)
        EXTERNAL_FUNCTION(cos)
        EXTERNAL_FUNCTION(tan)
        EXTERNAL_FUNCTION(sinh)
        EXTERNAL_FUNCTION(cosh)
        EXTERNAL_FUNCTION(tanh)
        EXTERNAL_FUNCTION(asin)
        EXTERNAL_FUNCTION(acos)
        EXTERNAL_FUNCTION(atan)
        EXTERNAL_FUNCTION(abs)
        EXTERNAL_FUNCTION(sqrt)
        EXTERNAL_FUNCTION(log)
        EXTERNAL_FUNCTION(sq)
        EXTERNAL_FUNCTION(cb)
        EXTERNAL_FUNCTION(cbrt)

#undef EXTERNAL_FUNCTION

        detail::printable_type short_print(const accumulator_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,true);
            return ostr.str();
        }

        detail::printable_type short_print(const result_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,true);
            return ostr.str();
        }

        detail::printable_type full_print(const accumulator_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,false);
            return ostr.str();
        }

        detail::printable_type full_print(const result_wrapper& arg)
        {
            std::ostringstream ostr;
            arg.print(ostr,false);
            return ostr.str();
        }
        
    }
}
