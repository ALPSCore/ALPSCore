/***************************************************************************
* ALPS++/lattice library
*
* lattice/point_traits.h     default point traits
*
* $Id$
*
* Copyright (C) 2001-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*                            Synge Todo <wistaria@comp-phys.org>
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

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
