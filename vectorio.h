/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_VECTORIO_H
#define ALPS_VECTORIO_H

#include <alps/config.h>
#include <alps/parser/parser.h>
#include <alps/vectortraits.h>

#include <boost/throw_exception.hpp>
#include <iostream>
#include <string>
#ifndef BOOST_NO_STRINGSTREAM
# include <sstream>
#else
# include <strstream>
# define istringstream istrstream
# define ostringstream ostrstream
#endif
#include <stdexcept>

namespace alps {

template <class CONTAINER>
inline void read_vector_resize (std::istream& in, CONTAINER& v)
{
  typedef typename VectorTraits<CONTAINER>::value_type value_type;
  std::vector<value_type> tmp;
  while(true) {
    value_type x;
    in >> x;
    if (!in)
      break;
    tmp.push_back(x);
  }
  vectorops::resize(v,tmp.size());
  for (int i=0;i<vectorops::size(v);++i)
    v[i]=tmp[i];
}

template <class CONTAINER>
inline void read_vector (std::istream& in, CONTAINER& v)
{
  typename VectorTraits<CONTAINER>::size_type i=0;
  while (i!=vectorops::size(v))
    in >> v[i++];
}

template <class CONTAINER>
inline void read_vector (std::istream& in, CONTAINER& v, 
         typename VectorTraits<CONTAINER>::size_type dim)
{
  vectorops::resize(v,dim);
  read_vector(in,v);
}

template <class CONTAINER>
inline void read_vector (const std::string& s, CONTAINER& v,
         typename VectorTraits<CONTAINER>::size_type dim)
{
  std::istringstream in(s.c_str());
  read_vector(in,v,dim);
}

template <class CONTAINER>
inline void read_vector (const std::string& s, CONTAINER& v)
{
  std::istringstream in(s.c_str());
  read_vector(in,v);
}

template <class CONTAINER>
inline void read_vector_resize (const std::string& s, CONTAINER& v)
{
  std::istringstream in(s.c_str());
  read_vector_resize(in,v);
}

template <class CONTAINER>
inline CONTAINER read_vector (const std::string& s,
         typename VectorTraits<CONTAINER>::size_type dim)
{
  CONTAINER v;
  read_vector(s,v,dim);
  return v;
}

template <class CONTAINER>
inline CONTAINER read_vector (const std::string& s)
{
  CONTAINER v;
  read_vector(s,v);
  return v;
}

template <class CONTAINER>
inline CONTAINER read_vector (std::istream& in,
         typename VectorTraits<CONTAINER>::size_type dim)
{
  CONTAINER v;
  read_vector(in,v,dim);
  return v;
}

template <class CONTAINER>
inline CONTAINER read_vector (std::istream& in)
{
  CONTAINER v;
  read_vector(in,v);
  return v;
}

template <class CONTAINER>
inline void write_vector(std::ostream& out, const CONTAINER& v)
{
  for (std::size_t i=0;i<vectorops::size(v);++i) {
    out << v[i];
    if (i!=vectorops::size(v)-1)
      out << " ";
  }
}

template <class CONTAINER>
inline std::string vector_writer(const CONTAINER& c)
{
  std::ostringstream str;
  write_vector(str,c);
  return str.str();
}

} // end namespace alps

#endif // ALPS_VECTORIO_H
