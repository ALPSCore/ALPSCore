/***************************************************************************
* PALM++/osiris library
*
* osiris/std/impl.h      dumps for object serialization
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
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

#ifndef OSIRIS_STD_IMPL_H
#define OSIRIS_STD_IMPL_H

// #include <palm/config.h>
#include <alps/osiris/dump.h>

#include <algorithm>
#include <complex>

namespace alps {
namespace detail {

/// function object for serialization
template<class T>
class saveTo {
  alps::ODump& dump_;
public:
  /// the constructor takes the ODump as argument
  saveTo(alps::ODump& dump) : dump_(dump) {}
  /// the function call operator serializes the object
  inline void operator()(const T& x) { dump_ << x;}
};
  
/// serialize a container
template<class C>
inline void saveContainer(alps::ODump& dump, const C& x)
{
  saveTo<typename C::value_type> save(dump);
  dump << uint32_t(x.size());
  std::for_each(x.begin(),x.end(),save);
}

/// deserialize a set-like container
template<class C>
inline void loadContainer(alps::IDump& dump, C& x)
{
  x=C();
  uint32_t n(dump);
  typename C::value_type val;
  while (n--) {
    dump >> val;
    x.push_back(val);
  }
}

/// deserialize a list-like container
template<class C>
inline alps::IDump& loadArrayLikeContainer(alps::IDump& dump, C& x)
{
  x.resize(uint32_t(dump));
  for(typename C::iterator p=x.begin();p!=x.end();++p)
    dump >> *p;
  return dump;
}

/// deserialize a stack-like container
template<class C>
inline alps::IDump& loadStackLikeContainer(alps::IDump& dump, C& x)
{
  x=C(); // empty stack
  int n(dump); // number of elements
  
  typename C::ValueType elem;
  while (n--)
    {
      dump >> elem;
      x.push(elem);
    }
  return dump;
}

/// deserialize a stack-like container
template<class C>
inline alps::IDump& loadSetLikeContainer(alps::IDump& dump, C& x)
{
  x=C(); // empty stack
  int n(dump); // number of elements

  typename C::ValueType elem;
  while (n--)
    {
      dump >> elem;
      x.insert(elem,x.end());
    }
  return dump;
}


/** traits class specifying for whoch types there are optimized
    saveArray functions in the dump classes */

template <class T> struct TypeDumpTraits {
  static const bool hasArrayFunction=false;
};

#define _HAS_ARRAY(T) template<> struct TypeDumpTraits< T > {\
  static const bool hasArrayFunction=true;};
  
#ifdef PALM_HAS_INT64
_HAS_ARRAY(int64_t)
_HAS_ARRAY(uint64_t)
#endif
_HAS_ARRAY(int32_t)
_HAS_ARRAY(int16_t)
_HAS_ARRAY(int8_t)
_HAS_ARRAY(uint32_t)
_HAS_ARRAY(uint16_t)
_HAS_ARRAY(uint8_t)
_HAS_ARRAY(float)
_HAS_ARRAY(double)
_HAS_ARRAY(long double)
_HAS_ARRAY(bool)
_HAS_ARRAY(std::complex<float>)
_HAS_ARRAY(std::complex<double>)
_HAS_ARRAY(std::complex<long double>)

#undef _HAS_ARRAY

} // end namespace detail
} // end namespace alps

#endif // OSIRIS_std_IMPL_H
