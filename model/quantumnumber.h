/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2004 by Matthias Troyer <troyer@comp-phys.org>,
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
#include <alps/parameters.h>
#include <alps/expression.h>
#include <boost/lexical_cast.hpp>
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

  bool valid(value_type x) const { return x >= min() && x<= max(); }
  const std::string min_expression() const { return min_string_; }
  const std::string max_expression() const { return max_string_; }
  value_type min() const
  {
    if (!valid_ && !evaluate())
      boost::throw_exception(std::runtime_error("Cannot evaluate expression " +
                                                min_string_ ));
    return min_;
  }
  value_type max() const
  {
    if (!valid_ && !evaluate())
      boost::throw_exception(std::runtime_error("Cannot evaluate expression " +
                                                max_string_ ));
    return max_;
  }
  I levels() const
  {
    return (max().distance(min())==std::numeric_limits<I>::max()) ?
      std::numeric_limits<I>::max() : (max().distance(min()) + 1);
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
{}

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
{}


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
    if (min()!=value_type::min() && rhs.min()!=value_type::min())
      min_ += rhs.min_;
    if (max()!=value_type::max() && rhs.max()!=value_type::max())
      max_ += rhs.max_;
  }
  if (fermionic() != rhs.fermionic())
    boost::throw_exception(std::runtime_error("Adding fermionic and bosonic quantum numbers: " + name() + " + " + rhs.name()));
return *this;
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
}

template <class I>
bool QuantumNumberDescriptor<I>::set_parameters(const Parameters& p)
{
  return evaluate(p);
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
    min_ = value_type::min();
  else if (min_exp_.can_evaluate(eval))
#ifndef ALPS_WITH_NEW_EXPRESSION
    min_ = alps::evaluate(min_exp_);
#else
    min_ = alps::evaluate<double>(min_exp_);
#endif
  else valid_=false;
  if (max_exp_=="infinity")
    max_ = value_type::max();
  else if (max_exp_.can_evaluate(eval))
#ifndef ALPS_WITH_NEW_EXPRESSION
    max_ = alps::evaluate(max_exp_);
#else
    max_ = alps::evaluate<double>(max_exp_);
#endif
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
