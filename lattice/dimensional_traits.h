/***************************************************************************
* ALPS++/lattice library
*
* lattice/dimensional_traits.h     default dimensional traits
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

#ifndef ALPS_LATTICE_DIMENSIONAL_TRAITS_H
#define ALPS_LATTICE_DIMENSIONAL_TRAITS_H

#include <alps/config.h>
#include <boost/limits.hpp>

namespace alps {

template <class Dimensional>
struct dimensional_traits {
  typedef std::size_t dimension_type;
  static const bool fixed_dimension=false;
  static dimension_type infinity()
  {
    return std::numeric_limits<dimension_type>::max();
  }
};

template <class Dimensional>
inline typename dimensional_traits<Dimensional>::dimension_type
dimension(const Dimensional& d)
{
  return d.size();
}

/*
template <class T, int sz>
struct dimensional_traits<T[sz]> {
  typedefint dimension_type;
  static const bool fixed_dimension=true;
  static const dimension_type dimension=sz;
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
