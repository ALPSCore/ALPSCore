/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_IS_COMPLEX_H
#define ALPS_TYPE_TRAITS_IS_COMPLEX_H

#include <complex>
#include <type_traits>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct is_complex : std::false_type {};

template <class T>
struct is_complex<std::complex<T> > : std::true_type {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_IS_COMPLEX_H
