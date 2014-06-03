/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef OSIRIS_std_MAP_H
#define OSIRIS_std_MAP_H

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/impl.h>
#include <alps/osiris/std/pair.h>

#include <map>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// deserialize a std::map container
template <class Key, class T, class Compare, class Allocator>
inline alps::IDump& operator >> (alps::IDump& dump,
                                   std::map<Key,T,Compare,Allocator>& x)
{
  x=std::map<Key,T,Compare,Allocator>();
  uint32_t n(dump);
  Key k;
  T v;
  while (n--)
    {
      dump >> k >> v;
      x[k]=v;
    }
  
  return dump;
}

/// serialize a std::map container
template <class Key, class T, class Compare, class Allocator>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const std::map<Key,T,Compare,Allocator>& x)
{
  alps::detail::saveContainer(dump,x);
  return dump;
}          

/// deserialize a std::multimap container
template <class Key, class T, class Compare, class Allocator>
inline alps::IDump& operator >> (alps::IDump& dump,
                                   std::multimap<Key,T,Compare,Allocator>& x)
{
  return alps::detail::loadSetLikeContainer(dump,x);
}

                          
/// serialize a std::multimap container
template <class Key, class T, class Compare, class Allocator>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const std::multimap<Key,T,Compare,Allocator>& x)
{
  return alps::detail::saveContainer(dump,x);
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_std_MAP_H
