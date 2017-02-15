/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_PARAMS_TYPEINDEX_HPP_INCLUDED
#define ALPS_PARAMS_TYPEINDEX_HPP_INCLUDED

/** @file typeindex.hpp
    @brief Provides human-readable type names for error messages
    @warning boost::TypeIndex does the same job better, but is not available until Boost 1.56.0.
*/

#include <typeinfo>
#include "boost/preprocessor/stringize.hpp"

namespace alps {
    namespace params_ns {
        namespace detail {

            /// Default generic implementation for any type name
            template <typename T>
            class type_id {
                public:
                /// Generic: uses typeid(T).name()
                static const char* pretty_name()
                {
                    return typeid(T).name();
                }
            };

            // /// Type names for vectors (NOTE: will not work correctly for vectors of vectors)
            // template <typename T>
            // class type_id< std::vector<T> > {
            //     /// Generic: uses typeid(T).name()
            //     static const char* pretty_name()
            //     {
            //         return typeid(T).name();
            //     }
            // };

                
// Macro for generating pretty names of chosen types
#define ALPS_PARAMS_DETAIL_TYPID_NAME(atype)    \
            template <>                         \
            class type_id<atype> {              \
              public:                           \
                static const char* pretty_name()\
                {                               \
                    return BOOST_PP_STRINGIZE(atype); \
                }                                     \
            };        

        } // detail
    } // params_ns
} // alps
#endif // ALPS_PARAMS_TYPEINDEX_HPP_INCLUDED
