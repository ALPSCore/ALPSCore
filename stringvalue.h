/***************************************************************************
* ALPS++ library
*
* parser/stringvalue.h     StringValue class: a class which can take 
*                          any value of a few standard data types
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_STRINGVALUE_H
#define ALPS_STRINGVALUE_H

#include <alps/config.h>
#include <boost/lexical_cast.hpp>
#include <complex>
#include <iostream>
#include <string>

namespace alps {

class StringValue
{
public:
  StringValue() {}
  StringValue(const StringValue& x) : value_(x.value_) {}
  StringValue(const std::string& x) : value_(x) {}
  StringValue(const char * x) : value_(x) {}
  
  template <class T>
  StringValue(const T& x) : value_(boost::lexical_cast<std::string, T>(x)) {}

  operator std::string () const { return value_;}

  bool operator==(const StringValue& x) const {return value_==x.value_; }
  bool operator!=(const StringValue& x) const {return value_!=x.value_; }

  template <class T>
  T get() const { return operator T(); }
    
  operator bool() const { 
    return value_ == "true" ? true : (value_ == "false" ? false : boost::lexical_cast<bool,std::string>(value_));
  }

#define CONVERTIT(A) operator A() const { return boost::lexical_cast<A, std::string>(value_); }
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

#endif // ALPS_PARSER_STRINGVALUE_H
