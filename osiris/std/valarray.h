/***************************************************************************
* PALM++/osiris library
*
* osiris/std/valarray.h      serialize a std::valarray
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

#ifndef OSIRIS_STD_VALARRAY_H
#define OSIRIS_STD_VALARRAY_H

#include <alps/config.h>

#ifdef HAVE_VALARRAY

#include <alps/osiris/dump.h>
#include <alps/osiris/std/impl.h>
#include <valarray>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// deserialize a std::valarray container
template <class T>
inline alps::IDump& operator >> (alps::IDump& dump, std::valarray<T>& x)
{
  x.resize(uint32_t(dump));
  dump.read_array(x.size(),&(x[0]));
  return dump;
}

/// serialize a std::valarray container
template <class T>
inline alps::ODump& operator << (alps::ODump& dump,
				   const std:: valarray<T>& x)
{
  dump << uint32_t(x.size());
  dump.write_array(x.size(),&(const_cast<std::valarray<T>&>(x)[0]));
  return dump;
}	  

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // HAVE_VALARRAY

#endif // OSIRIS_STD_VALARRAY_H
