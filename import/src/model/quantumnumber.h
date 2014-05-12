/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2009 by Matthias Troyer <troyer@comp-phys.org>,
*                            Axel Grzesik <axel@th.physik.uni-bonn.de>,
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

#ifndef ALPS_MODEL_QUANTUMNUMBER_H
#define ALPS_MODEL_QUANTUMNUMBER_H

#include <alps/model/half_integer.h>
#include <alps/parser/xmlstream.h>
#include <alps/parser/parser.h>
#include <alps/parameter.h>
#include <alps/expression.h>
#include <boost/config.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/optional.hpp>
#include <boost/tuple/tuple.hpp>
#include <set>
#include <stdexcept>
#include <string>

namespace alps {

template<class I>
class QuantumNumberDescriptor
{
public:
  typedef half_integer<I> value_type;
  QuantumNumberDescriptor(const std::string& n, value_type minVal=0, value_type maxVal=0,
                bool f=false);
  QuantumNumberDescriptor(const std::string& n, const std::string& min_str,
                const std::string& max_str, bool f=false);
  QuantumNumberDescriptor(const XMLTag&, std::istream&);

  bool valid(value_type x) const
  {
    return (x >= min BOOST_PREVENT_MACRO_SUBSTITUTION ()) &&
      (x <= max BOOST_PREVENT_MACRO_SUBSTITUTION ()); 
  }

  const std::string min_expression() const { return min_string_; }
  const std::string max_expression() const { return max_string_; }
  
  value_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const
  {
    if (!valid_ && !evaluate())
      boost::throw_exception(std::runtime_error("Cannot evaluate expression " +
                                                min_string_ + "in QuantumNumberDescriptor::min()"));
    return min_;
  }
  
  value_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const
  {
    if (!valid_ && !evaluate())
      boost::throw_exception(std::runtime_error("Cannot evaluate expression " +
                                                max_string_  + "in QuantumNumberDescriptor::max()"));
    return max_;
  }
  
  value_type global_max() const 
  {
    return global_max_ ? global_max_.get() : max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }

  value_type global_min() const 
  {
    return global_min_ ? global_min_.get() : min BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }

  value_type global_increment() const 
  {
    return increment_;
  }
  
  typedef boost::tuple<value_type,value_type,value_type> range_type;
  range_type global_range() const 
  { 
    return boost::make_tuple(global_min(),global_max(),global_increment());
  }
  
  I levels() const
  {
    return (max BOOST_PREVENT_MACRO_SUBSTITUTION ().distance(min BOOST_PREVENT_MACRO_SUBSTITUTION ())==std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) ?
      std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION () : (max BOOST_PREVENT_MACRO_SUBSTITUTION ().distance(min BOOST_PREVENT_MACRO_SUBSTITUTION ()) + 1);
  }
  
  const std::string& name() const { return name_; }

  const QuantumNumberDescriptor& operator+=(const QuantumNumberDescriptor& rhs);

  void write_xml(alps::oxstream&) const;
  bool fermionic() const { return fermionic_;}

  bool set_parameters(const Parameters&);
  // returns true if it can be evaluated

  bool depends_on(const Parameters::key_type& s) const;
  bool depends_on(const QuantumNumberDescriptor& qn) const
  { return (dependency_.find(qn)!=dependency_.end()); }
  void add_dependency(const QuantumNumberDescriptor& qn) { dependency_.insert(qn); }

  void reset_limits()
  {
    global_min_.reset();
    global_max_.reset();
    increment_=1;
  }

  void update_limits()
  {
    value_type m = min BOOST_PREVENT_MACRO_SUBSTITUTION ();
    if (global_min_) {
      if (global_min_.get().is_even() != m.is_even())
        increment_=0.5;
      if (m < global_min_.get())
        global_min_ = m;
    }
    else
      global_min_ = m;
    m = max BOOST_PREVENT_MACRO_SUBSTITUTION ();
    if (global_max_) {
      if (global_max_.get().is_even() != m.is_even())
        increment_=0.5;
      if (m > global_max_.get())
      global_max_ = m;
    }
    else
      global_max_ = m;
  }

private:
  std::string name_;
  std::string min_string_;
  std::string max_string_;
  mutable value_type min_;
  mutable value_type max_;
  bool fermionic_;
  mutable bool valid_;
  bool evaluate(const Parameters& =Parameters()) const;
  mutable std::set<QuantumNumberDescriptor> dependency_;
  boost::optional<value_type> global_min_;
  boost::optional<value_type> global_max_;
  value_type increment_;
};

template<class I>
inline bool operator< (const QuantumNumberDescriptor<I>& q1,const QuantumNumberDescriptor<I>& q2) {
  return q1.name()<q2.name();
}

template <class I>
QuantumNumberDescriptor<I>:: QuantumNumberDescriptor(const std::string& n, value_type minVal, value_type maxVal, bool f)
   : name_(n),
     min_string_(boost::lexical_cast<std::string,value_type>(minVal)),
     max_string_(boost::lexical_cast<std::string,value_type>(maxVal)),
     min_(minVal),
     max_(maxVal),
     fermionic_(f),
     valid_(true)
{
  reset_limits();
}

template <class I>
QuantumNumberDescriptor<I>:: QuantumNumberDescriptor(const std::string& n,
                                 const std::string& min_str,
                                 const std::string& max_str,
                                 bool f)
   : name_(n),
     min_string_(min_str),
     max_string_(max_str),
     min_(),
     max_(),
     fermionic_(f),
     valid_(true)
{
  reset_limits();
}


template <class I>
const QuantumNumberDescriptor<I>& QuantumNumberDescriptor<I>::operator+=(const QuantumNumberDescriptor<I>& rhs)
{

  Parameters p;
  if(dependency_.size()!=rhs.dependency_.size())
    boost::throw_exception(std::runtime_error("Adding quantum numbers that do not depend on the same quantum numbers: " + name() + " + " + rhs.name()));
  for(typename std::set<QuantumNumberDescriptor<I> >::const_iterator it=dependency_.begin();it!=dependency_.end();++it) {
    if(!rhs.depends_on(*it)) boost::throw_exception(std::runtime_error("Adding quantum numbers that do not both depend on quantum number " + it->name() + ": " + name() + " + " + rhs.name()));
    p[it->name()]=0;
  }

  ParameterEvaluator eval(p);
  Expression min_exp(min_string_);
  Expression max_exp(max_string_);
  min_exp.partial_evaluate(eval);
  max_exp.partial_evaluate(eval);
  min_string_ = boost::lexical_cast<std::string>(min_exp+Expression(rhs.min_string_));
  max_string_ = boost::lexical_cast<std::string>(max_exp+Expression(rhs.max_string_));
  if (valid_) {
    if (min BOOST_PREVENT_MACRO_SUBSTITUTION () != 
        value_type::min BOOST_PREVENT_MACRO_SUBSTITUTION () &&
        rhs.min BOOST_PREVENT_MACRO_SUBSTITUTION () !=
        value_type::min BOOST_PREVENT_MACRO_SUBSTITUTION ())
      min_ += rhs.min_;
    if (max BOOST_PREVENT_MACRO_SUBSTITUTION () !=
        value_type::max BOOST_PREVENT_MACRO_SUBSTITUTION () &&
        rhs.max BOOST_PREVENT_MACRO_SUBSTITUTION () !=
        value_type::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
      max_ += rhs.max_;
  }
  if (fermionic() != rhs.fermionic())
    boost::throw_exception(std::runtime_error("Adding fermionic and bosonic quantum numbers: " + name() + " + " + rhs.name()));
return *this;
  reset_limits();
}

template <class I>
QuantumNumberDescriptor<I> operator+(const QuantumNumberDescriptor<I>& x,const QuantumNumberDescriptor<I>& y)
{
  QuantumNumberDescriptor<I> res(x);
  res +=y;
  return res;
}

template <class I>
QuantumNumberDescriptor<I>::QuantumNumberDescriptor(const XMLTag& intag, std::istream&)
 : valid_(false)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  fermionic_ = tag.attributes["type"]=="fermionic";
  min_string_=tag.attributes["min"];
  if (min_string_=="")
    boost::throw_exception(std::runtime_error("min attribute missing in QUANTUMNUMBER element"));
  max_string_=tag.attributes["max"];
  if (max_string_=="")
    boost::throw_exception(std::runtime_error("max attribute missing in QUANTUMNUMBER element"));
  reset_limits();
}

template <class I>
bool QuantumNumberDescriptor<I>::set_parameters(const Parameters& p)
{
  bool could_evaluate = evaluate(p);
  if (could_evaluate)
    update_limits();
  else
    reset_limits();
  return could_evaluate;
}

template<class I >
bool QuantumNumberDescriptor<I>::depends_on(const Parameters::key_type& s) const
{
  Expression min_exp_(min_string_);
  Expression max_exp_(max_string_);
  return (min_exp_.depends_on(s) || max_exp_.depends_on(s));
}

template <class I>
bool QuantumNumberDescriptor<I>::evaluate(const Parameters& p) const
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
    min_ = value_type::min BOOST_PREVENT_MACRO_SUBSTITUTION ();
  else if (min_exp_.can_evaluate(eval))
    min_ = alps::evaluate<double>(min_exp_);
  else valid_=false;
  if (max_exp_=="infinity")
    max_ = value_type::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  else if (max_exp_.can_evaluate(eval))
    max_ = alps::evaluate<double>(max_exp_);
  else valid_=false;
  if(valid_ && min_>max_)
    boost::throw_exception(std::runtime_error("min > max in QUANTUMNUMBER element"));
  return valid_;
}

template <class I>
void QuantumNumberDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("QUANTUMNUMBER") << attribute("name", name())
     << attribute("min", min_expression()) << attribute("max", max_expression());
  if (fermionic())
    os << attribute("type","fermionic");
   os << end_tag("QUANTUMNUMBER");
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::QuantumNumberDescriptor<I>& q)
{
  q.write_xml(out);
  return out;
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::QuantumNumberDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
