/***************************************************************************
* ALPS++/lattice library
*
* lattice/lattice.h    the lattice class
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

#ifndef ALPS_LATTICE_LATTICE_H
#define ALPS_LATTICE_LATTICE_H

#include <alps/config.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/coordinate_traits.h>
#include <alps/vectorio.h>

namespace alps {

template <class L>
struct lattice_traits
{
};

template <class Lattice>
inline const typename lattice_traits<Lattice>::unit_cell_type&
unit_cell(const Lattice& l)
{
  return l.unit_cell();
}

template <class Lattice>
inline typename lattice_traits<Lattice>::cell_descriptor
cell(const typename lattice_traits<Lattice>::offset_type& o,const Lattice& l)
{
  return l.cell(o);
}

template <class Lattice>
inline const typename lattice_traits<Lattice>::offset_type&
offset(const typename lattice_traits<Lattice>::cell_descriptor& c, const Lattice& )
{
  return c.offset();
}

template <class Lattice>
inline typename lattice_traits<Lattice>::size_type
volume(const Lattice& l)
{
  return l.volume();
}

template <class Lattice>
inline bool 
on_lattice(typename lattice_traits<Lattice>::offset_type o, const Lattice& l)
{
  return l.on_lattice(o);
}

template <class Lattice>
inline std::pair<typename lattice_traits<Lattice>::cell_iterator,
                 typename lattice_traits<Lattice>::cell_iterator>
cells(const Lattice& l)
{
  return l.cells();
}

template <class Lattice>
inline std::pair<bool, typename lattice_traits<Lattice>::boundary_crossing_type>
shift(typename lattice_traits<Lattice>::offset_type& o,
      const typename lattice_traits<Lattice>::offset_type& s,
      const Lattice& l)
{
  return l.shift(o,s);
}

template <class Lattice>
inline typename lattice_traits<Lattice>::size_type
index(typename lattice_traits<Lattice>::cell_descriptor c, const Lattice& l)
{
  return l.index(c);
}

template <class Lattice>
inline std::pair<typename lattice_traits<Lattice>::basis_vector_iterator,
                 typename lattice_traits<Lattice>::basis_vector_iterator>
basis_vectors(const Lattice& l)
{
  return l.basis_vectors();
}

template <class Lattice>
inline typename lattice_traits<Lattice>::vector_type
coordinate(const typename lattice_traits<Lattice>::cell_descriptor& c, 
       const typename lattice_traits<Lattice>::vector_type& p, const Lattice& l)
{
  typename lattice_traits<Lattice>::basis_vector_iterator first, last;
  typedef typename coordinate_traits<typename lattice_traits<Lattice>::offset_type
    >::const_iterator offset_iterator;
  boost::tie(first,last) = basis_vectors(l);
  offset_iterator off = coordinates(offset(c,l)).first; 
  typename lattice_traits<Lattice>::vector_type v(alps::dimension(*first));
  for (int i=0; first!=last; ++first, ++off,++i)
    v = v + (*first) * ((*off)+(p.size() ? p[i] : 0));
  return v;
}
    
template <class Lattice>
inline typename lattice_traits<Lattice>::vector_type
origin(const typename lattice_traits<Lattice>::cell_descriptor& c, const Lattice& l)
{
  typename lattice_traits<Lattice>::basis_vector_iterator first, last;
  typedef typename coordinate_traits<typename lattice_traits<Lattice>::offset_type
    >::const_iterator offset_iterator;
  boost::tie(first,last) = basis_vectors(l);
  offset_iterator off = coordinates(offset(c,l)).first; 
  typename lattice_traits<Lattice>::vector_type v;
  for (; first!=last; ++first, ++off)
    v = v + (*first) * (*off);
  return v;
}
  void prevent_optimization();

} // end namespace alps

#endif // ALPS_LATTICE_LATTICE_H
