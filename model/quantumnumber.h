/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003 by Matthias Troyer <troyer@comp-phys.org>,
*                       Axel Grzesik <axel@th.physik.uni-bonn.de>,
*                       Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_MODEL_QUANTUMNUMBER_H
#define ALPS_MODEL_QUANTUMNUMBER_H

#include <alps/config.h>
#ifdef ALPS_WITHOUT_XML
#error "Model library needs XML support"
#endif

#include <alps/parser/xmlstream.h>
#include <alps/parser/parser.h>
#include <alps/parameters.h>
#include <alps/expression.h>
#include <boost/lexical_cast.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <cassert>

namespace alps {

template <class I>
class half_integer {
public:
  typedef I integer_type;
  half_integer() : val_(0) {}
  half_integer(double x) :val_(integer_type(2*x+(x<0?-0.01:0.01))) {}
  half_integer& operator=(const half_integer& x) {
    val_ = x.val_;
    return *this;
  }
  half_integer& operator=(double x) {
    val_=integer_type(2*x+(x < 0 ? -0.01 : 0.01));
    return *this;
  }
  operator double() const { return 0.5*val_; }
  
  void set_half(integer_type x) { val_=x; }
  integer_type get_twice() const { return val_; }
  
  template <class J> bool operator==(const half_integer<J>& rhs) const
  { return val_ == rhs.val_; }
  template <class J> bool operator!=(const half_integer<J>& rhs) const
  { return val_ != rhs.val_; }
  template <class J> bool operator<(const half_integer<J>& rhs) const
  { return val_ < rhs.val_; }
  template <class J> bool operator>(const half_integer<J>& rhs) const
  { return val_ > rhs.val_; }
  template <class J> bool operator<=(const half_integer<J>& rhs) const
  { return val_ <= rhs.val_; }
  template <class J> bool operator>=(const half_integer<J>& rhs) const
  { return val_ >= rhs.val_; }

  half_integer operator-() const { return half_integer(-val_, 0); }

  half_integer& operator++() { val_ += 2; return *this; }
  half_integer& operator--() { val_ -= 2; return *this; }
  half_integer operator++(int)
    { half_integer tmp(*this); ++(*this); return tmp; }
  half_integer operator--(int)
    { half_integer tmp(*this); --(*this); return tmp; }

  half_integer& operator+=(integer_type x) { val_ += 2*x; return *this; }
  half_integer& operator-=(integer_type x) { val_ -= 2*x; return *this; }
  template <class J>
  half_integer& operator+=(const half_integer<J>& x)
  { val_ += x.val_; return *this; }
  template <class J>
  half_integer& operator-=(const half_integer<J>& x)
  { val_ -= x.val_; return *this; }

  template <class J>
  half_integer operator+(const half_integer<J>& x) const
  { half_integer res(*this); return res += x; }
  template <class J>
  half_integer operator-(const half_integer<J>& x) const
  { half_integer res(*this); return res -= x; }
  half_integer operator+(integer_type x) const
  { half_integer res(*this); return res += x; }
  half_integer operator-(integer_type x) const
  { half_integer res(*this); return res -= x; }

  integer_type distance(const half_integer& x) const
  { 
    if ((*this==max()) != (x==max())) return std::numeric_limits<I>::max();
    if (std::numeric_limits<I>::is_signed && (*this==min())!=(x==min()))
      return std::numeric_limits<I>::max();
    assert(std::abs(val_)%2 == std::abs(x.val_)%2);
    return (val_-x.val_)/2;
  }
  static half_integer max()
  { return half_integer(std::numeric_limits<I>::max(),0); }
  static half_integer min()
  {
    return std::numeric_limits<I>::is_signed ? 
      -half_integer(std::numeric_limits<I>::max(),0) :
      half_integer(std::numeric_limits<I>::min(),0);
  }
  
private:
  half_integer(integer_type i, int /* to distinguish */) : val_(i) {}
  integer_type val_;
};

template <class I>
inline half_integer<I> operator+(I x, const half_integer<I>& y)
{ return y + x; }

template <class I>
inline half_integer<I> operator-(I x, const half_integer<I>& y)
{ return - y + x; }

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
  QuantumNumber(const std::string& n, value_type minVal=0, value_type maxVal=0, bool f=false);
#ifndef ALPS_WITHOUT_XML
  QuantumNumber(const XMLTag&, std::istream&);
#endif
  
  bool valid(value_type x) const { return x >= min() && x<= max();}
  const std::string min_expression() const { return min_string_;}
  const std::string max_expression() const { return max_string_;}
  value_type min() const 
  {if (!valid_ && !evaluate()) boost::throw_exception(std::runtime_error("Cannot evaluate expression " + min_string_ )); return _min;} 
  value_type max() const 
  {if (!valid_ && !evaluate()) boost::throw_exception(std::runtime_error("Cannot evaluate expression " + max_string_ )); return _max;} 
  I levels() const {
    return (max().distance(min())==std::numeric_limits<I>::max()) ? std::numeric_limits<I>::max() : max().distance(min())+1;}
  const std::string& name() const {return _name;}
  //bool operator== ( const QuantumNumber& x) const
  //{ return min()==x.min() && max() ==x.max() && name() == x.name();} 
 
  const QuantumNumber& operator+=(const QuantumNumber& rhs);

  void write_xml(alps::oxstream&) const;
  bool fermionic() const { return _fermionic;}
  bool set_parameters(const Parameters&); // returns true if it can be evaluated
  bool depends_on(const Parameters::key_type& s) const;
  bool depends_on(const QuantumNumber& qn) const { return (dependency_.find(qn)!=dependency_.end()); }
  void add_dependency(const QuantumNumber& qn) { dependency_.insert(qn); }
private:
  std::string _name;
  std::string min_string_;
  std::string max_string_;
  mutable value_type _min;
  mutable value_type _max;
  bool _fermionic;
  mutable bool valid_;
  bool evaluate(const Parameters& =Parameters()) const;
  mutable std::set<QuantumNumber> dependency_;
};

template<class I>
inline bool operator< (const QuantumNumber<I>& q1,const QuantumNumber<I>& q2) {
  return q1.name()<q2.name();
}

template <class I>
QuantumNumber<I>:: QuantumNumber(const std::string& n, value_type minVal, value_type maxVal, bool f)
   : _name(n), 
     _min(minVal),
     _max(maxVal),
     _fermionic(f), 
     valid_(true), 
     min_string_(boost::lexical_cast<std::string,value_type>(minVal)),
     max_string_(boost::lexical_cast<std::string,value_type>(maxVal))
{}


template <class I>
const QuantumNumber<I>& QuantumNumber<I>::operator+=(const QuantumNumber<I>& rhs)
{
  Parameters p; 
  if(dependency_.size()!=rhs.dependency_.size())
    boost::throw_exception(std::runtime_error("Adding quantum numbers that do not depend on the same quantum numbers: " + name() + " + " + rhs.name()));
  for(typename std::set<QuantumNumber<I> >::const_iterator it=dependency_.begin();it!=dependency_.end();++it) {
    if(!rhs.depends_on(*it)) boost::throw_exception(std::runtime_error("Adding quantum numbers that do not both depend on quantum number " + it->name() + ": " + name() + " + " + rhs.name()));
    p[it->name()]=0;
  }
  
  ParameterEvaluator eval(p);
  Expression min_exp = Expression(min_string_);
  Expression max_exp = Expression(max_string_);
  min_exp.partial_evaluate(eval);
  max_exp.partial_evaluate(eval);
  min_string_ = static_cast<std::string>(min_exp+Expression(rhs.min_string_));
  max_string_ = static_cast<std::string>(max_exp+Expression(rhs.max_string_));
  if (valid_) {
    if (min()!=value_type::min() && rhs.min()!=value_type::min())
      _min += rhs._min;
    if (max()!=value_type::max() && rhs.max()!=value_type::max())
      _max += rhs._max;
  }
  if (fermionic() != rhs.fermionic())
    boost::throw_exception(std::runtime_error("Adding fermionic and bosonic quantum numbers: " + name() + " + " + rhs.name()));
return *this;
}

template <class I>
QuantumNumber<I> operator+(const QuantumNumber<I>& x,const QuantumNumber<I>& y)
{
  QuantumNumber<I> res(x);
  res +=y;
  return res;
}

#ifndef ALPS_WITHOUT_XML

template <class I>
QuantumNumber<I>::QuantumNumber(const XMLTag& intag, std::istream&)
 : valid_(false)
{
  XMLTag tag(intag);
  _name = tag.attributes["name"];
  _fermionic = tag.attributes["type"]=="fermionic";
  min_string_=tag.attributes["min"];
  if (min_string_=="")
    boost::throw_exception(std::runtime_error("min attribute missing in QUANTUMNUMBER element"));
  max_string_=tag.attributes["max"];
  if (max_string_=="")
    boost::throw_exception(std::runtime_error("max attribute missing in QUANTUMNUMBER element"));
}

template <class I>
bool QuantumNumber<I>::set_parameters(const Parameters& p)
{
  return evaluate(p);
}

template<class I >
bool QuantumNumber<I>::depends_on(const Parameters::key_type& s) const
{
  Expression min_exp_(min_string_);
  Expression max_exp_(max_string_);
  return (min_exp_.depends_on(s) || max_exp_.depends_on(s));
}

template <class I>
bool QuantumNumber<I>::evaluate(const Parameters& p) const
{
  ParameterEvaluator eval(p);
  Expression min_exp_(min_string_);
  Expression max_exp_(max_string_);
  min_exp_.partial_evaluate(eval);
  min_exp_.simplify();
  max_exp_.partial_evaluate(eval);
  max_exp_.simplify();
  valid_=true;
  if (min_exp_==" - infinity")
    _min = value_type::min();
  else if (min_exp_.can_evaluate(eval))
    _min = min_exp_.value();
  else valid_=false;
  if (max_exp_=="infinity")
    _max = value_type::max();
  else if (max_exp_.can_evaluate(eval))
    _max = max_exp_.value();
  else valid_=false;
  if(valid_ && _min>_max)
    boost::throw_exception(std::runtime_error("min > max in QUANTUMNUMBER element"));
  return valid_;
}
  
template <class I>
void QuantumNumber<I>::write_xml(oxstream& os) const
{
  os << start_tag("QUANTUMNUMBER") << attribute("name", name())
     << attribute("min", min_expression()) << attribute("max", max_expression());
  if (fermionic())
    os << attribute("type","fermionic");
   os << end_tag("QUANTUMNUMBER");
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::QuantumNumber<I>& q)
{
  q.write_xml(out);
  return out;        
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::QuantumNumber<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;        
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
