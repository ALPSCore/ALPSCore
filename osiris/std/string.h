/***************************************************************************
* PALM++/osiris library
*
* osiris/std/string.h      dumps for object serialization
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

#ifndef OSIRIS_STD_STRING_H
#define OSIRIS_STD_STRING_H

#include <alps/osiris/dump.h>

#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <string>

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

//=======================================================================
// string templates
//-----------------------------------------------------------------------

template <class charT, class traits, class Allocator>
inline alps::IDump& operator >> (alps::IDump& dump,
  std::basic_string<charT,traits,Allocator>& s)
{
  uint32_t sz(dump);
  if(sz) {
    charT* t = new charT[sz+1];
    dump.read_string(sz+1,t);
    if(t[sz]!=charT(0))
      boost::throw_exception(std::runtime_error("string on dump not terminating with '\\0'"));
	s=t;
    delete t;
    if(s.length()!=sz)
      boost::throw_exception(std::runtime_error("string on dump has incorrect length"));
  } else {
    s="";
  }
  return dump;
}

template <class charT, class traits, class Allocator>
inline alps::ODump& operator << (alps::ODump& dump,
  const std::basic_string<charT,traits,Allocator>& s)
{
 dump << uint32_t(s.size());
 if(s.size())
   dump.write_string(s.size()+1,s.c_str());
 return dump;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_STRING_H
