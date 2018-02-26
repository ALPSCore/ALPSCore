/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_IS_SEQUENCE_H
#define ALPS_TYPE_TRAITS_IS_SEQUENCE_H

#include <alps/config.hpp>
#include <alps/type_traits/has_value_type.hpp>
#include <type_traits>
#include <valarray>
#include <vector>
#include <complex>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct is_sequence : alps::has_value_type<T> {};

template <class T>
struct is_sequence<std::valarray<T> > : std::true_type {};

template <class T>
struct is_sequence<std::complex<T> > : std::false_type {};

template <>
struct is_sequence<std::string> : std::false_type {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_ELEMENT_TYPE_H
