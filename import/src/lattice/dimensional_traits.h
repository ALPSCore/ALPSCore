/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_DIMENSIONAL_TRAITS_H
#define ALPS_LATTICE_DIMENSIONAL_TRAITS_H

#include <alps/config.h>
#include <boost/limits.hpp>
#include <boost/config.hpp>

namespace alps {

template <class Dimensional>
struct dimensional_traits {
  typedef std::size_t dimension_type;
  BOOST_STATIC_CONSTANT(bool, fixed_dimension=false);
  static dimension_type infinity()
  {
    return std::numeric_limits<dimension_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }
};

/*
template <class Dimensional>
inline typename dimensional_traits<Dimensional>::dimension_type
dimension(const Dimensional& d)
{
  return d.size();
}
*/


template <class T, class A>
inline typename dimensional_traits<std::vector<T,A> >::dimension_type
dimension(const std::vector<T,A>& d)
{
  return d.size();
}

/*
template <class T, int sz>
struct dimensional_traits<T[sz]> {
  typedefint dimension_type;
  BOOST_STATIC_CONSTANT(bool, fixed_dimension=true);
  BOOST_STATIC_CONSTANT(dimension_type, dimension=sz);
  static dimension_type infinity() { return=std::numeric_limits<dimension_type>::max();}
};
  
template <class T, int sz>
typename dimensional_traits<T[sz]>::dimension_type
inline dimension(const T[sz]& d)
{
  return sz;
}
*/

} // end namespace alps

#endif // ALPS_LATTICE_DIMENSIONAL_TRAITS_H
