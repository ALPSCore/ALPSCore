/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_ACCUMULATOR_ARGUMENTS_HPP
#define ALPS_NGS_ALEA_ACCUMULATOR_ARGUMENTS_HPP

#include <boost/parameter.hpp>

// = = = = N A M E D   P A R A M E T E R   D E F I N I T I O N = = = =

namespace alps {
    namespace accumulator  {

        BOOST_PARAMETER_NAME((bin_size, keywords) _bin_size)
        BOOST_PARAMETER_NAME((bin_num, keywords) _bin_num)
        BOOST_PARAMETER_NAME((weight_ref, keywords) _weight_ref)
        BOOST_PARAMETER_NAME((Weight, keywords) _Weight)

    } // end accumulator namespace
} // end alps namespace
#endif // ALPS_NGS_ALEA_ACCUMULATOR_ARGUMENTS_HPP
