/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2011 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef OSIRIS_BOOST_UBLAS_H
#define OSIRIS_BOOST_UBLAS_H

// #include <palm/config.h>
#include <boost/numeric/ublas/vector.hpp>
#include <alps/osiris/std/impl.h>

/// deserialize a boost::numeric::ublas::vector container

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class T, class STORAGE>
inline alps::IDump& operator >> (alps::IDump& dump,
                                   boost::numeric::ublas::vector<T,STORAGE>& x)
{
  x.resize(uint32_t(dump));
  if (x.size())
    dump.read_array(x.size(),&(x[0]));
  return dump;
}

/// serialize a boost::numeric::ublas::vector container
template <class T, class STORAGE>
inline alps::ODump& operator << (alps::ODump& dump,
                                   const boost::numeric::ublas::vector<T,STORAGE>& x)
{
  dump << uint32_t(x.size());
  if(x.size())
    dump.write_array(x.size(),&(x[0]));
  return dump;
}          

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // OSIRIS_BOOST_UBLAS_H
