/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_HAS_VALUE_TYPE_H
#define ALPS_TYPE_TRAITS_HAS_VALUE_TYPE_H

#include <type_traits>

namespace alps {

template<typename T> struct has_value_type {
    template<typename U> static char check(typename U::value_type *);
    template<typename U> static double check(...);
    typedef std::integral_constant<bool, sizeof(char) == sizeof(check<T>(0))> type;
    constexpr static bool value = type::value;
};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_HAS_VALUE_TYPE_H
