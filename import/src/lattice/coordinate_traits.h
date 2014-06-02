/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_LATTICE_COORDINATE_TRAITS_H
#define ALPS_LATTICE_COORDINATE_TRAITS_H

#include <alps/config.h>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <valarray>

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


template <class C>
std::string coordinate_to_string(const C& c, int precision = 0)
{
  std::ostringstream str;
  str << "( ";
  if (precision > 0) str << std::setprecision(precision);
  int n=0;
  typename coordinate_traits<C>::const_iterator first, last;
  for (boost::tie(first,last) = coordinates(c); first != last; ++first, ++n) {
    if (n) str << ',';
    str << *first;
  }
  str << " )";
  return str.str();
} 

} // end namespace alps

#endif // ALPS_LATTICE_COORDINATE_TRAITS_H
