/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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
