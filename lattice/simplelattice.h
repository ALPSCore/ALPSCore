/***************************************************************************
* ALPS++/lattice library
*
* lattice/simplelattice.h    the lattice class
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.org>
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

#ifndef ALPS_LATTICE_SIMPLELATTICE_H
#define ALPS_LATTICE_SIMPLELATTICE_H

#include <alps/config.h>
#include <alps/lattice/cell_traits.h>
#include <alps/lattice/dimensional_traits.h>
#include <alps/lattice/coordinate_traits.h>
#include <alps/lattice/simplecell.h>
#include <alps/lattice/lattice.h>

namespace alps {

template <class UnitCell=EmptyUnitCell, class Cell = simple_cell<UnitCell> >
class simple_lattice {
public:
  typedef UnitCell unit_cell_type;
  typedef Cell cell_descriptor;
  typedef typename alps::dimensional_traits<unit_cell_type>::dimension_type dimension_type;
  typedef typename cell_traits< cell_descriptor>::offset_type offset_type;
  
  simple_lattice() {}

  template <class U2, class C2>
  simple_lattice(const simple_lattice<U2,C2>& l)
   : unit_cell_(alps::unit_cell(l))
  {    std::cerr << "simple_lattice\n";}
  
  template <class U2, class C2>
  const simple_lattice& operator=(const simple_lattice<U2,C2>& l)
  {
    unit_cell_ = alps::unit_cell(l);
    return *this;
  }

  const unit_cell_type& unit_cell() const { return unit_cell_;}
  
  cell_descriptor cell(offset_type o) const 
  { return cell_descriptor(unit_cell(),o); }
  
  dimension_type dimension() const { return alps::dimension(unit_cell_);}
protected:
  unit_cell_type unit_cell_;
};

template <class U, class C>
inline typename dimensional_traits<simple_lattice<U,C> >::dimension_type
dimension (const simple_lattice<U,C>& l)
{
  return l.dimension();
}

template <class UnitCell, class Cell>
struct lattice_traits<simple_lattice<UnitCell,Cell> >
{
  typedef typename simple_lattice<UnitCell,Cell>::unit_cell_type unit_cell_type;
  typedef typename simple_lattice<UnitCell,Cell>::cell_descriptor cell_descriptor;
  typedef typename simple_lattice<UnitCell,Cell>::offset_type offset_type;
};

} // end namespace alps

#endif // ALPS_LATTICE_SIMPLELATTICE_H
