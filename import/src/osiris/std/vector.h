/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_STD_VECTOR_H
#define OSIRIS_STD_VECTOR_H

// #include <palm/config.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/impl.h>

#include <vector>

/// deserialize a std::vector container

namespace alps {
namespace detail {

template <class T, bool OPTIMIZED> struct VectorHelper {};

template <class T> struct VectorHelper<T,false> {
  template <class ALLOCATOR>
  static void read(alps::IDump& dump, std::vector<T,ALLOCATOR>& x) 
  {
    loadArrayLikeContainer(dump,x);
  }
  template <class ALLOCATOR>
  static void write(alps::ODump& dump, const std::vector<T,ALLOCATOR>& x) 
  {
    saveContainer(dump,x);
  }
};

template <class T> struct VectorHelper<T,true> {
  template <class ALLOCATOR>
  static void read(alps::IDump& dump, std::vector<T,ALLOCATOR>& x) 
  {
    x.resize(uint32_t(dump));
    if (x.size())
      dump.read_array(x.size(),&(x[0]));
  }
  
  template <class ALLOCATOR>
  static void write(alps::ODump& dump, const std::vector<T,ALLOCATOR>& x) 
  {
    dump << uint32_t(x.size());
    if(x.size())
      dump.write_array(x.size(),&(x[0]));
  }
};

} // end namespace detail
} // end namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T, class Allocator>
inline alps::IDump& operator >> (alps::IDump& dump,
                                   std::vector<T,Allocator>& x)
{
  alps::detail::VectorHelper<T,alps::detail::TypeDumpTraits<T>::hasArrayFunction>::read(dump,x);
  return dump;
}

/// serialize a std::vector container
template <class T, class Allocator>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const std:: vector<T,Allocator>& x)
{
  alps::detail::VectorHelper<T,alps::detail::TypeDumpTraits<T>::hasArrayFunction>::write(dump,x);
  return dump;
}          

/// serialize a std::vector<bool> from compressed form
template <class Allocator>
inline alps::IDump& operator >> (alps::IDump& dump,
                                   std::vector<bool,Allocator>& x)
{
  // map from integer Array
  x.resize(uint32_t(dump));
  uint32_t words=(x.size()+31)/32;
  std::vector<uint32_t> tmp(words);
  dump.read_array(words,&(tmp[0]));
  for (size_t i=0;i<x.size();i++)
    x[i] = (tmp[i/32]&(1<<(i%32)));
  return dump;
}

/// serialize a std::vector<bool> in compressed form
template <class Allocator>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const std:: vector<bool,Allocator>& x)
{
  //  to integer Array
  uint32_t n=x.size();
  uint32_t words=(n+31)/32;
  std::vector<uint32_t> tmp(words);
  for (size_t i=0;i<n;i++)
    if(x[i])
      tmp[i/32] |= 1<<(i%32);
  dump << n;
  dump.write_array(words,&(tmp[0]));
  return dump;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_VECTOR_H
