/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_IS_SCALAR_H
#define ALPS_TYPE_TRAITS_IS_SCALAR_H

#include <alps/type_traits/is_complex.hpp>
#include <boost/type_traits/is_scalar.hpp>
#include <boost/mpl/or.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct is_scalar : public boost::mpl::or_<boost::is_scalar<T>,is_complex<T> > {};

template <class T>
struct is_scalar<std::complex<T> > : public is_scalar<T> {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_IS_SCALAR_H
