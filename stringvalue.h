/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Mathias Koerner <mkoerner@itp.phys.ethz.ch>
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

#ifndef ALPS_STRINGVALUE_H
#define ALPS_STRINGVALUE_H

#include <alps/config.h>
#include <boost/lexical_cast.hpp>
#include <complex>
#include <iostream>
#include <string>
#include <stdexcept>

#define NEW_STRINGVALUE_IMPL 1

#ifndef NEW_STRINGVALUE_IMPL

namespace alps {

class StringValue
{
public:
  StringValue() {}
  StringValue(const std::string& x) : value_(x) {}
  StringValue(const char * x) : value_(x) {}

  template <class T>
  StringValue(const T& x) : value_(boost::lexical_cast<std::string, T>(x)) {}
  // we "recast" errors below to std::runtime_error, but here we
  // leave it as a boost::bad_lexical_cast?! (mkoerner)

  operator std::string () const { return value_;}

  bool operator==(const StringValue& x) const {return value_==x.value_; }
  bool operator!=(const StringValue& x) const {return value_!=x.value_; }

  template <class T>
  T get() const { return operator T(); }

  operator bool() const {
    return value_ == "true" ? true : (value_ == "false" ? false : boost::lexical_cast<bool,std::string>(value_));
  }

#ifdef BOOST_NO_EXCEPTIONS
#define CONVERTIT(A) operator A() const { return boost::lexical_cast<A, std::string>(value_); }
#else
#define CONVERTIT(A) operator A() const { \
  try { return boost::lexical_cast<A, std::string>(value_);} \
  catch (...) { boost::throw_exception(std::runtime_error(\
  "Could not convert string '"+value_+"to type "#A));}\
  return boost::lexical_cast<A, std::string>(value_);}
#endif
  CONVERTIT(int8_t)
  CONVERTIT(uint8_t)
  CONVERTIT(int16_t)
  CONVERTIT(uint16_t)
  CONVERTIT(int32_t)
  CONVERTIT(uint32_t)
#ifndef BOOST_NO_INT64_T
  CONVERTIT(int64_t)
  CONVERTIT(uint64_t)
#endif
  CONVERTIT(float)
  CONVERTIT(double)
  CONVERTIT(long double)
  CONVERTIT(std::complex<float>)
  CONVERTIT(std::complex<double>)
  CONVERTIT(std::complex<long double>)
#undef CONVERTIT

  const char* c_str() const { return value_.c_str();}

  const StringValue& operator=(const std::string& x) {
    value_ = x;
    return *this;
  }
  const StringValue& operator=(const char* x) {
    value_ = x;
    return *this;
  }
  template <class T>
  const StringValue& operator=(const T& x) {
    value_ = boost::lexical_cast<std::string, T>(x);
    return *this;
  }

  bool valid() const {return !value_.empty(); }

private:
  std::string value_;
};

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator << (std::ostream& os, const alps::StringValue& v)
{
  return os << static_cast<std::string>(v);
}

inline std::istream& operator >> (std::istream& is, alps::StringValue& v)
{
  std::string s;
  is >> s;
  v = s;
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#else // NEW_STRINGVALUE_IMPL

namespace alps {

// New implementation for the ValueString class. By deriving from
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
typedef lexical_cast_string<> StringValue;

namespace detail {

template<class S, class T>
struct lexical_cast_string_helper
{
  static T get(const lexical_cast_string<S>& s)
  { return boost::lexical_cast<T>(s); }
};

template<class S>
struct lexical_cast_string_helper<S, S>
{
  static S get(const lexical_cast_string<S>& s) { return s; }
};

} // end namespace detail

template<class StringBase>
class lexical_cast_string : public StringBase
{
public:
  typedef StringBase base_type;

  lexical_cast_string(const base_type& s = base_type()) : base_type(s) {}
  lexical_cast_string(const lexical_cast_string& s) : base_type(s) {}
  lexical_cast_string(const char* s) : base_type(s) {}
  template <class T>
  lexical_cast_string(const T& x)
    : base_type(boost::lexical_cast<base_type>(x)) {}

  bool valid() const { return !empty(); }

  template <class T> T get() const
  { return detail::lexical_cast_string_helper<base_type, T>::get(*this); }

  operator bool() const {
    if ( *this == "true" ) return true;
    if ( *this == "false" ) return false;
    return boost::lexical_cast<bool>(*this);
  }

#define CONVERTIT(T) operator T() const { \
  return boost::lexical_cast<T>(*this); \
}
  CONVERTIT(int8_t)
  CONVERTIT(uint8_t)
  CONVERTIT(int16_t)
  CONVERTIT(uint16_t)
  CONVERTIT(int32_t)
  CONVERTIT(uint32_t)
#ifndef BOOST_NO_INT64_T
  CONVERTIT(int64_t)
  CONVERTIT(uint64_t)
#endif
  CONVERTIT(float)
  CONVERTIT(double)
  CONVERTIT(long double)
  CONVERTIT(std::complex<float>)
  CONVERTIT(std::complex<double>)
  CONVERTIT(std::complex<long double>)
#undef CONVERTIT

};

} // end namespace alps

#endif // NEW_STRINGVALUE_IMPL

#endif // ALPS_PARSER_STRINGVALUE_H
