/***************************************************************************
* ALPS++/lattice library
*
* lattice/coordinate_traits.h     default lattice traits
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

#ifndef ALPS_LATTICE_COORDINATE_TRAITS_H
#define ALPS_LATTICE_COORDINATE_TRAITS_H

#include <alps/config.h>
#include <algorithm>

#ifdef HAVE_VALARRAY
#include <valarray>
#endif

namespace alps {

template <class C>
struct coordinate_traits {
  typedef typename C::value_type value_type;
  typedef typename C::iterator iterator;
  typedef typename C::const_iterator const_iterator;
};
  
template <class C>
struct coordinate_traits<const C> {
  typedef typename C::value_type value_type;
  typedef typename C::const_iterator iterator;
  typedef typename C::const_iterator const_iterator;
};
  
template <class C>
inline std::pair<typename coordinate_traits<C>::iterator, typename coordinate_traits<C>::iterator>
coordinates(C& c)
{
  return std::make_pair(c.begin(),c.end());
}

// template <class C>
// inline std::pair<typename coordinate_traits<C>::const_iterator, typename coordinate_traits<C>::const_iterator>
// coordinates(const C& c)
// {
//   return std::make_pair(c.begin(),c.end());
// }

template <class T, int sz>
struct coordinate_traits<T[sz]> {
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
};
  
/*
template <class T, int sz>
inline std::pair<T*, T*>
coordinates(T[sz]& c)
{
  return std::make_pair(c,c+sz);
}

template <class T, int sz>
inline std::pair<const T*, const T*>
coordinates(const T[sz]& c)
{
  return std::make_pair(c,c+sz);
}
*/

#ifdef HAVE_VALARRAY
template <class T>
struct coordinate_traits<std::valarray<T> > {
  typedef T value_type;
  typedef T* iterator;
  typedef const T* const_iterator;
};
  
template <class T>
inline std::pair<T*, T*>
coordinates(std::valarray<T>& c)
{
  return make_pair(&(c[0]),&(c[0])+c.size());
}

template <class T>
inline std::pair<const T*, const T*>
coordinates(const std::valarray<T>& c)
{
  return std::pair<const T*, const T*>
    (&(const_cast<std::valarray<T>&>(c)[0]),
    &(const_cast<std::valarray<T>&>(c)[0])+c.size());
}
#endif

} // end namespace alps

#endif // ALPS_LATTICE_COORDINATE_TRAITS_H
