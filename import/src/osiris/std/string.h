/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

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


inline alps::IDump& operator >> (alps::IDump& dump, std::string& s)
{
  dump.read_string(s);
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

inline alps::ODump& operator << (alps::ODump& dump, const std::string& s)
{
  dump.write_string(s);
  return dump;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_STD_STRING_H
