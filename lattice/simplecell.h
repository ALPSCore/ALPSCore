/***************************************************************************
* ALPS++/lattice library
*
* lattice/simplecell.h     simple cell implementation
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

#ifndef ALPS_LATTICE_SIMPLECELL_H
#define ALPS_LATTICE_SIMPLECELL_H

#include <alps/config.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/graph_traits.h>
#include <alps/lattice/dimensional_traits.h>
#include <alps/lattice/cell_traits.h>
#include <alps/vectorio.h>

namespace alps {

template <class UnitCell=EmptyUnitCell, class Offset=typename std::vector<int> >
class simple_cell  {
public:
  typedef Offset offset_type;
  typedef UnitCell unit_cell_type;
  typedef typename alps::dimensional_traits<UnitCell>::dimension_type dimension_type;

  simple_cell(const unit_cell_type& u, const offset_type& o)
   : dim_(alps::dimension(u)), offset_(o) {}
  
  const offset_type& offset() const { return offset_;}
  dimension_type dimension() { return dim_;}
private:
  dimension_type dim_;
  offset_type offset_;
};

template <class UnitCell,class Offset>
inline typename simple_cell<UnitCell,Offset>::dimension_type
dimension(const simple_cell<UnitCell,Offset>& c)
{
  return c.dimension();
}

} // end namespace alps

#endif // ALPS_LATTICE_SIMPLECELL_H
