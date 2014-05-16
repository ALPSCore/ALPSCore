/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_NORM_TYPE_H
#define ALPS_TYPE_TRAITS_NORM_TYPE_H

#include <alps/type_traits/real_type.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct norm_type  : public real_type<T> {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_NORM_TYPE_H
