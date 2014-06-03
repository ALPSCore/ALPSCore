/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

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
