/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

/// \file stringvalue.h
/// \brief implements a string class that can easily be assigned to and converted to any type

#ifndef ALPS_STRINGVALUE_H
#define ALPS_STRINGVALUE_H

#include <alps/config.h>
#include <boost/lexical_cast.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <stdexcept>


namespace alps {

// New implementation for the StringValue class. By deriving from
// std::string, we keep the complete functionality of std::string
// such as input and output, comparison, concatenation, while
// adding the conversion operators.
//
// The class is implemented by templating it on the
// basic string class we are using, although this might add
// some compilation overhead.
//
// The new implementation also does not catch any exceptions
// that occur, but instead lets any boost::bad_lexical_cast
// propagate.

template < class StringBase = std::string > class lexical_cast_string;

namespace detail {

/// a helper class to call lexical cast
template<class S, class T>
struct lexical_cast_string_helper
{
  static T get(const lexical_cast_string<S>& s)
  { return boost::lexical_cast<T>(s); }
};

/// \brief a helper class to call lexical cast, spexialized for std::string
///
/// it is a no-op for std::string
template<class S>
struct lexical_cast_string_helper<S, S>
{
  static S get(const lexical_cast_string<S>& s) { return s; }
};

} // end namespace detail

/// \brief a string class with built-in conversion to other types using lexical_cast
///
/// This class, derived from a std::string or similar implements additional
/// conversion operations and constructors implemented using boost::lexical_cast
/// \param StringBase the string class from which it is derived
template<class StringBase>
class lexical_cast_string : public StringBase
{
public:
  /// the underlying string class
  typedef StringBase string_type;

  /// constructor from a string
  lexical_cast_string(const string_type& s = string_type()) : string_type(s) {}
  /// copy-contructor
  lexical_cast_string(const lexical_cast_string& s) : string_type(s) {}
  /// constructor from a C-style string
  lexical_cast_string(const char* s) : string_type(s) {}
  /// constructor from a character sequence
  template<class InputItr>
  lexical_cast_string(InputItr first, InputItr last) 
#if BOOST_WORKAROUND(__IBMCPP__, <= 1200)
  {
    while (first!=last)
      (*this) += *first++;
  }
#else
  : string_type(first, last) {}
#endif

  /// constructor from arbitrary types implemented using boost::lexical_cast
  template <class T>
  lexical_cast_string(const T& x)
    : string_type(boost::lexical_cast<string_type>(x)) {}

  /// check whether the string is not empty
  bool valid() const { return !StringBase::empty(); }

  /// convert the string to type T using boost::lexical_cast
  template <class T> T get() const
  { return detail::lexical_cast_string_helper<string_type, T>::get(*this); }

  /// \brief convert the string to bool
  /// 
  /// the strings "true" and "false" are valid ways to specify tryue or false boolean values. Any other
  /// value will be converted to bool using boost::lexical_cast
  operator bool() const {
    if ( *this == "true" || *this == "True" ) return true;
    if ( *this == "false" || *this == "False" ) return false;
    return boost::lexical_cast<bool>(*this);
  }

/// INTERNAL ONLY
#define CONVERTIT(T) operator T() const {return boost::lexical_cast<T>(*this); }
  CONVERTIT(short)
  CONVERTIT(unsigned short)
  CONVERTIT(int)
  CONVERTIT(unsigned int)
  CONVERTIT(long)
  CONVERTIT(unsigned long)
#ifndef BOOST_NO_LONG_LONG
  CONVERTIT(long long)
  CONVERTIT(unsigned long long)
#endif
  /// convert the string to float
  CONVERTIT(float)
  /// convert the string to double
  CONVERTIT(double)
  /// convert the string to long double
  CONVERTIT(long double)
  /// convert the string to std::complex<float>
  CONVERTIT(std::complex<float>)
  /// convert the string to std::complex<double>
  CONVERTIT(std::complex<double>)
  /// convert the string to std::complex<long double>
  CONVERTIT(std::complex<long double>)
#undef CONVERTIT

};

/// StringValue is now implemented using lexical_cast_string
typedef lexical_cast_string<> StringValue;

} // end namespace alps

#endif // ALPS_PARSER_STRINGVALUE_H
