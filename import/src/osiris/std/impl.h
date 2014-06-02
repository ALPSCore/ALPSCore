/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

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
  
  typename C::value_type elem;
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

  typename C::value_type elem;
  while (n--)
    {
      dump >> elem;
      x.insert(x.end(),elem);
    }
  return dump;
}


/** traits class specifying for whoch types there are optimized
    saveArray functions in the dump classes */

template <class T> struct TypeDumpTraits {
  BOOST_STATIC_CONSTANT(bool, hasArrayFunction=false);
};

#define _HAS_ARRAY(T) template<> struct TypeDumpTraits< T > {\
  BOOST_STATIC_CONSTANT(bool, hasArrayFunction=true);};
  
#ifndef BOOST_NO_INT64_T
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
