/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_PYTHON_NUMPY_IMPORT_HPP
#define ALPS_NGS_PYTHON_NUMPY_IMPORT_HPP

#if defined(ALPS_HAVE_PYTHON)

    #include <alps/ngs/config.hpp>

    #include <alps/ngs/boost_python.hpp>

    #include <boost/python/numeric.hpp>

    #include <numpy/arrayobject.h>

    namespace alps {
        namespace detail {

            ALPS_DECL void import_numpy();

        }
    }

#endif

#endif
