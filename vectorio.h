/***************************************************************************
* ALPS++/lattice library
*
* lattice/vectorio.h     XML parser for lattices
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>
*                            Synge Todo <wistaria@comp-phys.org>
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
class vector_writer_t
{
public:
  vector_writer_t(const CONTAINER& c) : c_(c) {}
  void write(std::ostream& out) const {write_vector(out,c_);}
private:
  const CONTAINER& c_;
};

template <class CONTAINER>
inline vector_writer_t<CONTAINER> vector_writer(const CONTAINER& c)
{
  return vector_writer_t<CONTAINER>(c);
}

} // end namespace lattice

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class CONTAINER>
inline std::ostream& operator<<(std::ostream& out,
				const alps::vector_writer_t<CONTAINER>& w)
{
  w.write(out);
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif // ALPS_VECTORIO_H
