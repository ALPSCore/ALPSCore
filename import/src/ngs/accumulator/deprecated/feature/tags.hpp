/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_ALEA_DETAIL_TAU_HPP
#define ALPS_NGS_ALEA_DETAIL_TAU_HPP

namespace alps {
    namespace accumulator {
        // = = = = = = = = = = I M P L E M E N T A T I O N   D E F I N I T I O N = = = = = = = = = =
        namespace tag {
            struct mean;
            struct error;
            struct fixed_size_binning;
            struct max_num_binning;
            struct log_binning;
            struct autocorrelation;
            namespace detail {
                struct converged;
                struct tau;
                struct weight;
            }
            struct histogram;
        }
        namespace detail
        {
            struct no_weight_value_type {}; //has to be full type because of typeid
            //one cannot use void as wvt-default because of has_method in derived wrapper that checks if
            //there is an void op(vt, wvt) and void cannot be an argument-type
        }
    }
}
#endif
