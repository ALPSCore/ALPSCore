/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_LATTICE_H
#define ALPS_LATTICE_LATTICE_H

#include <alps/config.h>
#include <alps/lattice/unitcell.h>
#include <alps/lattice/coordinate_traits.h>

#include <alps/utility/vectorio.hpp>
#include <alps/numeric/vector_functions.hpp>
#include <alps/type_traits/element_type.hpp>

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
on_lattice(const typename lattice_traits<Lattice>::offset_type& o, const Lattice& l)
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
index(const typename lattice_traits<Lattice>::cell_descriptor& c, const Lattice& l)
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
inline std::pair<typename lattice_traits<Lattice>::basis_vector_iterator,
                 typename lattice_traits<Lattice>::basis_vector_iterator>
reciprocal_basis_vectors(const Lattice& l)
{
  return l.reciprocal_basis_vectors();
}


template <class Lattice>
inline typename lattice_traits<Lattice>::vector_type
coordinate(const typename lattice_traits<Lattice>::cell_descriptor& c, 
       const typename lattice_traits<Lattice>::vector_type& p, const Lattice& l)
{
  using boost::numeric::operators::operator+;
  using boost::numeric::operators::operator*;
  typename lattice_traits<Lattice>::basis_vector_iterator first, last;
  typedef typename coordinate_traits<typename lattice_traits<Lattice>::offset_type
    >::const_iterator offset_iterator;
  boost::tie(first,last) = basis_vectors(l);
  offset_iterator off = coordinates(offset(c,l)).first;
  typename lattice_traits<Lattice>::vector_type v(l.dimension());
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
  if (first!=last) {
    typename lattice_traits<Lattice>::vector_type v(*first);
    for (std::size_t d=0; d<v.size(); ++d)
      v[d]*=*off;
    ++first;
    ++off;
    for (; first!=last; ++first, ++off)
      for (std::size_t d=0; d<v.size(); ++d)
    v[d] += (*first)[d] * (*off);
    return v;
  }
  else
    return typename lattice_traits<Lattice>::vector_type();
}

void ALPS_DECL prevent_optimization();

template <class Lattice>
inline std::pair<typename lattice_traits<Lattice>::momentum_iterator,
                 typename lattice_traits<Lattice>::momentum_iterator>
momenta(const Lattice& l)
{
  return l.momenta();
}

template <class Lattice>
inline typename lattice_traits<Lattice>::vector_type
momentum(const typename lattice_traits<Lattice>::vector_type& m, const Lattice& l)
{
  typename lattice_traits<Lattice>::basis_vector_iterator first, last;
  boost::tie(first,last) = reciprocal_basis_vectors(l);
  if (first!=last) {
    typename lattice_traits<Lattice>::vector_type v(*first);
    for (std::size_t j=0; j<v.size(); ++j)
      v[j] *= m[0]/(2.*M_PI);
    ++first;
    for (int i=1; first!=last; ++first, ++i)
      for (std::size_t j=0; j<v.size(); ++j)
    v[j] = v[j] + (*first)[j] * m[i]/(2.*M_PI);
    return v;
  }
  else
    return m;
}

template <class Lattice>
inline typename lattice_traits<Lattice>::extent_type
extent(const Lattice& l)
{
  return l.extent();
}

template <class Lattice>
inline typename element_type<typename lattice_traits<Lattice>::extent_type>::type
extent(const Lattice& l, unsigned int d)
{
  return l.extent(d);
}

} // end namespace alps

#endif // ALPS_LATTICE_LATTICE_H
