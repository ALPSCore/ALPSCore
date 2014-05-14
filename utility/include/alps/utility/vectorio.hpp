/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file vectorio.h
/// \brief I/O helpers for vectors
///
/// This header contains helper functions to write and read vectors.
/// They are implemented based on the traits classes and functions in
/// vectortraits.h

#ifndef ALPS_VECTORIO_H
#define ALPS_VECTORIO_H

#include <alps/config.h>
//#include <alps/parser/parser.h> // FIXME
#include <alps/utility/size.hpp>
#include <alps/type_traits/element_type.hpp>

#include <boost/throw_exception.hpp>
#include <iomanip>
#include <iostream>
#include <iterator>
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

/// \brief reads a vector from a std::istream, until the end of the stream is reached.
/// \param in the stream
/// \param v the vector to be read
template <class CONTAINER>
inline void read_vector_resize (std::istream& in, CONTAINER& v)
{
  typedef typename element_type<CONTAINER>::type value_type;
  std::vector<value_type> tmp;
  std::copy(std::istream_iterator<value_type>(in),std::istream_iterator<value_type>(),std::back_inserter(tmp));
  v.resize(tmp.size());
  for (std::size_t i=0;i<size(v);++i)
    v[i]=tmp[i];
}

/// \brief reads a vector from a std::istream
/// \param in the stream
/// \param v the vector to be read
/// the number of elements to be read is taken from the size of the vector.
template <class CONTAINER>
inline void read_vector (std::istream& in, CONTAINER& v)
{
  std::size_t i=0;
  while (i!=alps::size(v))
    in >> v[i++];
}

/// \brief reads a vector from a std::istream
/// \param in the stream
/// \param v the vector to be read
/// \param n the number of elements to be read
template <class CONTAINER>
inline void read_vector (std::istream& in, CONTAINER& v, std::size_t n)
{
  v.resize(n);
  read_vector(in,v);
}


/// \brief reads a vector from a std::string, until the end of the string is reached.
/// \param s the string
/// \param v the vector to be read
template <class CONTAINER>
inline void read_vector_resize (const std::string& s, CONTAINER& v)
{
  std::istringstream in(s.c_str());
  read_vector_resize(in,v);
}

/// \brief reads a vector from a std::string, until the end of the string is reached.
/// \param s the string
/// \param v the vector to be read
/// the number of elements to be read is taken from the size of the vector.
template <class CONTAINER>
inline void read_vector (const std::string& s, CONTAINER& v)
{
  std::istringstream in(s.c_str());
  read_vector(in,v);
}

/// \brief reads a vector from a std::string
/// \param s the string
/// \param v the vector to be read
/// \param n the number of elements to be read
template <class CONTAINER>
inline void read_vector (const std::string& s, CONTAINER& v, std::size_t n)
{
  std::istringstream in(s.c_str());
  read_vector(in,v,n);
}


/// \brief reads a vector from a std::istream
/// \param in the stream
/// \param n the number of elements to be read
/// \return the vector to be read
template <class CONTAINER>
inline CONTAINER read_vector (std::istream& in, std::size_t n)
{
  CONTAINER v;
  read_vector(in,v,n);
  return v;
}

/// \brief reads a vector from a std::string, until the end of the string is reached.
/// \param s the string
/// \return the vector read from the string
template <class CONTAINER>
inline CONTAINER read_vector (const std::string& s)
{
  CONTAINER v;
  read_vector_resize(s,v);
  return v;
}

/// \brief reads a vector from a std::istream, until the end of the stream is reached.
/// \param in the stream
/// \return the vector read from the stream
template <class CONTAINER>
inline CONTAINER read_vector (std::istream& in)
{
  CONTAINER v;
  read_vector_resize(in,v);
  return v;
}

/// \brief reads a vector from a std::string
/// \param s the string
/// \param n the number of elements to be read
/// \return the vector read from the string
template <class CONTAINER>
inline CONTAINER read_vector (const std::string& s, std::size_t n)
{
  CONTAINER v;
  read_vector(s,v,n);
  return v;
}

/// \brief writes a vector to a std::string
/// \param v the vector to be written
/// \param delim delimitar between the elements of the vector
/// \param prec output precision of floating point elements
template <class CONTAINER>
inline std::string write_vector(const CONTAINER& v, const std::string& delim=" ", int prec = 20)
{
  std::ostringstream str;
  str << std::setprecision(prec);
  for (std::size_t i=0;i<alps::size(v);++i) {
    str << v[i];
    if (i!=alps::size(v)-1)
      str << delim;
  }
  return str.str();
}

} // end namespace alps

#endif // ALPS_VECTORIO_H
