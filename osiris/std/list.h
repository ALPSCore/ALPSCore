/***************************************************************************
* PALM++/osiris library
*
* osiris/std/list.h      dumps for object serialization
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

#ifndef OSIRIS_STD_LIST_H
#define OSIRIS_STD_LIST_H

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/impl.h>
#include <alps/osiris/std/pair.h>

#include <list>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// deserialize a std::list container
template <class T, class Allocator>
inline alps::IDump& operator>>(alps::IDump& dump, std::list<T,Allocator>& x)
{
  alps::detail::loadContainer(dump,x);
  return dump;
}

/// serialize a std::list container
template <class T, class Allocator>
inline alps::ODump& operator<<(alps::ODump& dump, const std::list<T,Allocator>& x)
{
  alps::detail::saveContainer(dump,x);
  return dump;
}	  

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_LIST_H
