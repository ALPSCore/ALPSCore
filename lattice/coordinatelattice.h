/***************************************************************************
* ALPS++/lattice library
*
* lattice/coordinatelattice.h    the lattice class
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
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

#ifndef ALPS_LATTICE_COORDINATELATTICE_H
#define ALPS_LATTICE_COORDINATELATTICE_H

#include <alps/config.h>
#include <alps/lattice/lattice.h>
#include <alps/lattice/simplelattice.h>
#include <alps/lattice/coordinate_traits.h>

#include <vector>

namespace alps {

template <class BASE = simple_lattice<>, class Vector = std::vector<double> >
class coordinate_lattice: public BASE {
public:
  typedef BASE parent_lattice_type;
  typedef typename lattice_traits<parent_lattice_type>::unit_cell_type unit_cell_type;
  typedef typename lattice_traits<parent_lattice_type>::offset_type offset_type;
  typedef typename lattice_traits<parent_lattice_type>::cell_descriptor cell_descriptor;
  typedef Vector vector_type;
  typedef typename std::vector<vector_type>::const_iterator basis_vector_iterator;
  
  coordinate_lattice() {}
  
  template <class B2,class V2>
  coordinate_lattice(const coordinate_lattice<B2,V2>& l)
   : parent_lattice_type(l),
     basis_vectors_(alps::basis_vectors(l).second-alps::basis_vectors(l).first)
  {
    typename lattice_traits<coordinate_lattice<B2,V2> >::basis_vector_iterator it;
    int i=0;
    for(it=alps::basis_vectors(l).first; it!=alps::basis_vectors(l).second;++it,++i)
      std::copy(it->begin(),it->end(),std::back_inserter(basis_vectors_[i]));
  }
  
  template <class InputIterator>
  coordinate_lattice(const unit_cell_type& u, InputIterator first, InputIterator last)
  : parent_lattice_type (u),
    basis_vectors_(first,last)
    {}

  coordinate_lattice(const unit_cell_type& u)
  : parent_lattice_type(u)
    {
    }

  template <class B2, class V2>
  const coordinate_lattice& operator=(const coordinate_lattice<B2,V2>& l)
  {
    static_cast<parent_lattice_type&>(*this)=l;
     basis_vectors_=std::vector<vector_type>(
       alps::basis_vectors(l).first, alps::basis_vectors(l).second);
     return *this;
  }

  std::pair<basis_vector_iterator,basis_vector_iterator>
  basis_vectors() const
  {
    return std::make_pair(basis_vectors_.begin(),basis_vectors_.end());
  }

protected:
  std::vector<vector_type> basis_vectors_;
};

template <class B, class V>
struct lattice_traits<coordinate_lattice<B,V> >
{
  typedef typename coordinate_lattice<B,V>::unit_cell_type unit_cell_type;
  typedef typename coordinate_lattice<B,V>::cell_descriptor cell_descriptor;
  typedef typename coordinate_lattice<B,V>::offset_type offset_type;
  typedef typename coordinate_lattice<B,V>::vector_type vector_type;
  typedef typename coordinate_lattice<B,V>::basis_vector_iterator basis_vector_iterator;
};

} // end namespace alps

#endif // ALPS_LATTICE_COORDINATELATTICE_H
