/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_IS_COMPLEX_H
#define ALPS_TYPE_TRAITS_IS_COMPLEX_H

#include <boost/mpl/bool.hpp>
#include <complex>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct is_complex : public boost::mpl::false_ {};

template <class T>
struct is_complex<std::complex<T> > : public boost::mpl::true_ {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_IS_COMPLEX_H
