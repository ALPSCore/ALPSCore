/***************************************************************************
* PALM++/osiris library
*
* osiris/std/pair.h      dumps for object serialization
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

#ifndef OSIRIS_STD_PAIR_H
#define OSIRIS_STD_PAIR_H

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <utility>

//=======================================================================
// pair templates
//-----------------------------------------------------------------------

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T1, class T2>
inline alps::IDump& operator>>(alps::IDump& dump, std::pair<T1,T2>& x)
{
  return dump >> x.first >> x.second;
}

template <class T1, class T2>
inline alps::ODump& operator<<(alps::ODump& dump, const std::pair<T1,T2>& x)
{
  return dump << x.first << x.second;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_PAIR_H
