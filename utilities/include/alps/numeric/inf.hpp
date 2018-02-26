/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NUMERIC_INF_HEADER
#define ALPS_NUMERIC_INF_HEADER

#include <alps/utilities/type_wrapper.hpp>

#include <limits>

namespace alps {
    namespace numeric {
        
        template<typename T> struct inf {};
        
#define ALPS_NUMERIC_INF_OVERLOADS(T)                                   \
        /** @brief Class convertible to infinity of type T */           \
        template<> struct inf< T > {                                    \
            /** @brief Unused argument, to adhere to the concept specifications */ \
            inf<T>(const T&) {}                                         \
                                                                        \
            /** @brief Returns infinity value */                        \
            operator T () const {                                       \
                return std::numeric_limits< T >::infinity();            \
            }                                                           \
        };
        ALPS_NUMERIC_INF_OVERLOADS(float)
        ALPS_NUMERIC_INF_OVERLOADS(double)
        ALPS_NUMERIC_INF_OVERLOADS(detail::type_wrapper<long double>::type)
#undef ALPS_NUMERIC_INF_OVERLOADS
        
    }
}

#endif
