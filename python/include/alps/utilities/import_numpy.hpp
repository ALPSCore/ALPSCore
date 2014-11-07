/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#pragma once

#include <alps/config.hpp>

#ifndef ALPS_HAVE_PYTHON
    #error numpy is only available if python is enabled
#endif

namespace alps {
    namespace detail {

        ALPS_DECL void import_numpy();
    }
}
