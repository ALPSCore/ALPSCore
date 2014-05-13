/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */


#ifndef ALPS_UTILITY_DATA_HPP
#define ALPS_UTILITY_DATA_HPP

#include <alps/type_traits/element_type.hpp>

namespace alps {

/// returns a pointer to the start of storage of a container
template <class C>
inline typename element_type<C>::type* data(C& c) { return &c[0];}

/// returns a pointer to the start of storage of a container
template <class C>
inline const typename element_type<C>::type* data(const C& c) { return &const_cast<C&>(c)[0];}

} // namespace alps

#endif // ALPS_UTILITY_DATA_HPP
