/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_CELL_TRAITS_H
#define ALPS_LATTICE_CELL_TRAITS_H

namespace alps {

template <class C>
struct cell_traits 
{
  typedef typename C::offset_type offset_type;
};

} // end namespace alps

#endif // ALPS_LATTICE_CELL_TRAITS_H
