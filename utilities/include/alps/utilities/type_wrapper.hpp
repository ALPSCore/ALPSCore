/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_DETAIL_TYPE_WRAPPER_HPP
#define ALPS_DETAIL_TYPE_WRAPPER_HPP

namespace alps {

    namespace detail {

        template<typename T> struct type_wrapper {
            typedef T type;
        };

    }
}
#endif
