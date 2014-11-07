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

// this must be first
#include <alps/utilities/boost_python.hpp>

#include <alps/utilities/import_numpy.hpp>

#include <complex>

#define ALPS_FOREACH_NATIVE_NUMPY_TYPE(CALLBACK)                                                                                                \
    CALLBACK(bool)                                                                                                                              \
    CALLBACK(char)                                                                                                                              \
    CALLBACK(signed char)                                                                                                                       \
    CALLBACK(unsigned char)                                                                                                                     \
    CALLBACK(short)                                                                                                                             \
    CALLBACK(unsigned short)                                                                                                                    \
    CALLBACK(int)                                                                                                                               \
    CALLBACK(unsigned)                                                                                                                          \
    CALLBACK(long)                                                                                                                              \
    CALLBACK(unsigned long)                                                                                                                     \
    CALLBACK(long long)                                                                                                                         \
    CALLBACK(unsigned long long)                                                                                                                \
    CALLBACK(float)                                                                                                                             \
    CALLBACK(double)                                                                                                                            \
    CALLBACK(long double)                                                                                                                       \
    CALLBACK(std::complex<float>)                                                                                                               \
    CALLBACK(std::complex<double>)                                                                                                              \
    CALLBACK(std::complex<long double>)

namespace alps {
    namespace detail {
        #define ALPS_DECL_NUMPY_TYPE(T)                                                                                                         \
            ALPS_DECL int get_numpy_type(T);
        ALPS_FOREACH_NATIVE_NUMPY_TYPE(ALPS_DECL_NUMPY_TYPE)
        #undef ALPS_DECL_NUMPY_TYPE
    }
}
