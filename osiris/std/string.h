/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

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
