/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_IS_SEQUENCE_H
#define ALPS_TYPE_TRAITS_IS_SEQUENCE_H

#include <alps/config.h>
#include <boost/mpl/bool.hpp>
#include <alps/type_traits/has_value_type.hpp>
#include <valarray>
#include <vector>
#include <complex>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct is_sequence : public alps::has_value_type<T> {};

template <class T>
struct is_sequence<std::valarray<T> > : public boost::mpl::true_ {};

template <class T>
struct is_sequence<std::complex<T> > : public boost::mpl::false_ {};
 
template <>
struct is_sequence<std::string> : public boost::mpl::false_ {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_ELEMENT_TYPE_H
