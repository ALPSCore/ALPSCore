/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_HAS_VALUE_TYPE_H
#define ALPS_TYPE_TRAITS_HAS_VALUE_TYPE_H

#include <boost/mpl/has_xxx.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

BOOST_MPL_HAS_XXX_TRAIT_DEF(value_type)

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_HAS_VALUE_TYPE_H
