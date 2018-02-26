/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_COVARIANCE_TYPE_H
#define ALPS_TYPE_TRAITS_COVARIANCE_TYPE_H

#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/element_type.hpp>
#include <alps/type_traits/is_sequence.hpp>

#include <type_traits>

namespace alps {

    namespace detail {
        template <typename T>
        struct matrix_covariance_type {
            matrix_covariance_type() {
                throw std::logic_error("Matrix covariance type is not implemented."
                                       " Use ALEA if you need covariance between vectors!");
            }
        };
    }

template <class T>
struct covariance_type
{
 typedef typename std::conditional<
     is_sequence<T>::value,
     detail::matrix_covariance_type<
       typename average_type<typename element_type<T>::type>::type
     >,
     typename average_type<T>::type
   >::type type;
};


} // end namespace alps

#endif // ALPS_TYPE_TRAITS_COVARIANCE_TYPE_H
