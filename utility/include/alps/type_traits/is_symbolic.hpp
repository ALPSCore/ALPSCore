/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: is_symbolic.hpp 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_IS_SYMBOLIC_H
#define ALPS_TYPE_TRAITS_IS_SYMBOLIC_H

#include <boost/mpl/bool.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct is_symbolic : public boost::mpl::false_ {};


} // end namespace alps

#endif // ALPS_TYPE_TRAITS_IS_SYMBOLIC_H
