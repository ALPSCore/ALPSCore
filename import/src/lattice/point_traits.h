/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_POINT_TRAITS_H
#define ALPS_LATTICE_POINT_TRAITS_H

namespace alps {

template <class P>
struct point_traits
{
  typedef P vector_type;        
};

} // end namespace alps

#endif // ALPS_LATTICE_POINT_TRAITS_H
