/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_SIMPLECELL_H
#define ALPS_LATTICE_SIMPLECELL_H

#include <alps/config.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/graph_traits.h>
#include <alps/lattice/dimensional_traits.h>
#include <alps/lattice/cell_traits.h>
#include <alps/utilities/vectorio.hpp>

namespace alps {

template <class UnitCell=EmptyUnitCell, class Offset=typename std::vector<int> >
class simple_cell  {
public:
  typedef Offset offset_type;
  typedef UnitCell unit_cell_type;
  typedef typename alps::dimensional_traits<UnitCell>::dimension_type dimension_type;

  simple_cell() : dim_(0) {}
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
