/***************************************************************************
* PALM++/osiris library
*
* osiris/std/map.h      dumps for object serialization
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
