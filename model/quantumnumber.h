/***************************************************************************
* ALPS++/model library
*
* model/quantumnumber.h    the quantumnumber classes
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>,
*                            Martin Joestingmeier
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

#ifndef ALPS_MODEL_QUANTUMNUMBER_H
#define ALPS_MODEL_QUANTUMNUMBER_H

#include <alps/config.h>

#ifndef ALPS_WITHOUT_XML
#include <alps/parser/parser.h>
#endif

#include <boost/lexical_cast.hpp>
#include <string>
#include <stdexcept>
#include <cassert>

namespace alps {

template <class I>
class half_integer {
public:
  typedef I integer_type;
  half_integer() : val_(0) {}
  half_integer(double x) :val_(integer_type(2*x+0.01)) {}
  const half_integer& operator=(double x) { val_=integer_type(2*x+0.01); return *this;}
  operator double() const { return 0.5*val_;}
  
  void set_half(integer_type x) { val_=x;}
  integer_type get_twice() const {return val_;}
  
  template <class J> bool operator==(half_integer<J> rhs) { return val_==rhs.val_;}
  template <class J> bool operator!=(half_integer<J> rhs) { return val_!=rhs.val_;}
  template <class J> bool operator<(half_integer<J> rhs) { return val_<rhs.val_;}
  template <class J> bool operator>(half_integer<J> rhs) { return val_>rhs.val_;}
  template <class J> bool operator<=(half_integer<J> rhs) { return val_<=rhs.val_;}
  template <class J> bool operator>=(half_integer<J> rhs) { return val_>=rhs.val_;}
  half_integer operator-() const { half_integer x(-val_,0); return x;}
  const half_integer& operator++() { val_+=2; return *this;}
  const half_integer& operator--() { val_-=2; return *this;}
  half_integer operator++(int) { half_integer tmp(*this); val_+=2; return tmp;}
  half_integer operator--(int) { half_integer tmp(*this); val_-=2; return tmp;}
  const half_integer& operator+=(integer_type x) { val_+=2*x; return *this;}
  const half_integer& operator-=(integer_type x) { val_-=2*x; return *this;}
  template <class J> const half_integer& operator+=(half_integer<J> x) { val_+=x.val_; return *this;}
  template <class J> const half_integer& operator-=(half_integer<J> x) { val_-=x.val_; return *this;}
  template <class J> half_integer operator+(half_integer<J> x) { half_integer res(*this); return res+=x;}
  template <class J> half_integer operator-(half_integer<J> x) { half_integer res(*this); return res-=x;}
  I distance(half_integer x) { assert(val_%2==x.val_%2); return (val_/2-x.val_/2);}
  static half_integer max() { return half_integer(std::numeric_limits<I>::max(),0);}
  static half_integer min() { return std::numeric_limits<I>::is_signed ? 
       -half_integer(std::numeric_limits<I>::max(),0): half_integer(std::numeric_limits<I>::min(),0);}
  
private:
  half_integer(integer_type i, int /* to distinguish */) : val_(i) {}
  integer_type val_;
};

template <class I>
inline std::ostream& operator<<(std::ostream& os, const half_integer<I>& x)
{
  if (x==half_integer<I>::max())
    return os << "infinity";
  else if (std::numeric_limits<I>::is_signed && x==half_integer<I>::min())
    return os << "-infinity";
  else if(x.get_twice() %2==0) 
    return os << x.get_twice()/2;
  else
    return os << x.get_twice() << "/2";
  return os;
}

template <class I>
inline std::istream& operator>>(std::istream& is, half_integer<I>& x)
{
  I nominator;
  is >> nominator;
  char c;
  is >> c;
  if ( is && c=='/') {
    is >> c;
    if (c!='2') {
      is.putback(c);
      is.putback('/');
      x.set_half(2*nominator);
    }
    x.set_half(nominator);
  }
  else {
    if (is)
      is.putback(c);
    x.set_half(2*nominator);
  }
  is.clear();
  return is;
}

template<class I>
class QuantumNumber
{
public:
  typedef half_integer<I> value_type;
  QuantumNumber(const std::string& n, value_type minVal, value_type maxVal, bool f=false)
   : _name(n), _min(minVal),_max(maxVal),_fermionic(f) {}
#ifndef ALPS_WITHOUT_XML
  QuantumNumber(const XMLTag&, std::istream&);
#endif
  
  bool valid(value_type x) const { return x >= min() && x<= max();}
  value_type min() const {return _min;} 
  value_type max() const {return _max;}
  I levels() const { return max().distance(min())+1;}
  const std::string& name() const {return _name;}
  bool operator== ( const QuantumNumber& x) const
  { return min()==x.min() && max() ==x.max() && name() == x.name();} 

#ifndef ALPS_WITHOUT_XML
  void write_xml(std::ostream&, const std::string& = "") const;
#endif
  bool fermionic() const { return _fermionic;}
private:
  std::string _name;
  value_type _min;
  value_type _max;
  bool _fermionic;
};

#ifndef ALPS_WITHOUT_XML

template <class I>
QuantumNumber<I>::QuantumNumber(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  _name = tag.attributes["name"];
  _fermionic = tag.attributes["type"]=="fermionic";
  if (tag.attributes["min"]=="-infinity")
    _min = value_type::min();
  else if (tag.attributes["min"]!="")
    _min = boost::lexical_cast<value_type,std::string>(tag.attributes["min"]);
  else
    boost::throw_exception(std::runtime_error("min attribute missing in QUANTUMNUMBER element"));
  if (tag.attributes["max"]=="infinity" || tag.attributes["max"]=="+infinity")
    _max = value_type::max();
  else if (tag.attributes["max"]!="")
    _max = boost::lexical_cast<value_type,std::string>(tag.attributes["max"]);
  else
    boost::throw_exception(std::runtime_error("max attribute missing in QUANTUMNUMBER element"));
  if(_min>_max)
    boost::throw_exception(std::runtime_error("min > max in QUANTUMNUMBER element"));
}

template <class I>
void QuantumNumber<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<QUANTUMNUMBER name=\"" << name() 
    << "\" min=\"" << min() <<  "\" max=\"" << max() << "\"";
  if (fermionic())
    os << " type=\"fermionic\"";
   os << "/>\n";
}

#endif

} // namespace alps

#ifndef ALPS_WITHOUT_XML

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::QuantumNumber<I>& q)
{
  q.write_xml(out);
  return out;	
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif

#endif
