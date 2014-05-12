/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2010 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_EXPRESSION_FACTOR_H
#define ALPS_EXPRESSION_FACTOR_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/symbol.h>
#include <alps/expression/number.h>
#include <alps/expression/block.h>
#include <alps/expression/function.h>
#include <alps/expression/evaluator.h>
#include <alps/type_traits/norm_type.hpp>
#include <boost/call_traits.hpp>

namespace alps {
namespace expression {

template<class T>
class SimpleFactor : public Evaluatable<T> {
public:
  typedef T value_type;
  typedef typename alps::norm_type<T>::type norm_type;
  
  SimpleFactor(std::istream&);
  SimpleFactor(typename boost::call_traits<value_type>::param_type x)
    : term_(new Number<T>(x)) {}
  SimpleFactor(const std::string& s) : term_(new Symbol<T>(s)) {}

  SimpleFactor(const SimpleFactor& v)
    : Evaluatable<T>(v), term_()
  {
    if (v.term_) term_.reset(v.term_->clone());
  }
  
  SimpleFactor(const Evaluatable<T>& v) : Evaluatable<T>(v), term_(v.clone()) {}
  virtual ~SimpleFactor() {}

  const SimpleFactor& operator=(const SimpleFactor& v);

  value_type value(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  void output(std::ostream&) const;
  bool can_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  Evaluatable<T>* clone() const { return new SimpleFactor<T>(*this); }
  void partial_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false);
  bool is_single_term() const { return term_ ? term_->is_single_term() : false; }
  Term<T> term() const { return term_ ? term_->term() : Term<T>(); }
  bool depends_on(const std::string& s) const
  {
    return term_ ? term_->depends_on(s) : false;
  }

protected:
  boost::shared_ptr<Evaluatable<T> > term_;
};

template<class T>
class Factor : public SimpleFactor<T> {
public:
  typedef T value_type;
  typedef SimpleFactor<T> super_type;
  typedef typename alps::norm_type<T>::type norm_type;
  
  Factor(std::istream&, bool inverse = false);
  Factor(typename boost::call_traits<value_type>::param_type x)
    : super_type(x), is_inverse_(false), power_(1.) {}
  Factor(const std::string& s) : super_type(s), is_inverse_(false), power_(1.) {}
  Factor(const Evaluatable<T>& v) : super_type(v), is_inverse_(false), power_(1.) {}
  Factor(const super_type& v) : super_type(v), is_inverse_(false), power_(1.) {}
  virtual ~Factor() {}
  value_type value(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  void output(std::ostream&) const;
  bool can_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  Evaluatable<T>* clone() const { return new Factor<T>(*this); }
  boost::shared_ptr<Factor> flatten_one_value();
  bool is_inverse() const { return is_inverse_; }
  void partial_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false);
  Term<T> term() const { return unit_power() ? super_type::term() : (super_type::term_ ? Term<T>(*this) : Term<T>()); }
  bool depends_on(const std::string& s) const
  {
    return super_type::depends_on(s) || power_.depends_on(s);
  }
  bool unit_power() const { return power_.can_evaluate() && power_.value() ==1.;}
  bool is_single_term() const { return super_type::is_single_term() && unit_power(); }

private:
  bool is_inverse_;
  SimpleFactor<T> power_;
};

//
// implementation of Factor<T>
//

template<class T>
SimpleFactor<T>::SimpleFactor(std::istream& in) : term_()
{
  char c;
  in >> c;

  // read value
  if (std::isdigit(c) || c=='.' || c=='+' || c=='-') {
    in.putback(c);
    typename Number<T>::real_type val;
    in >> val;
    if (!in)
      boost::throw_exception(std::runtime_error("Failed to parse number in factor"));
    term_.reset(new Number<T>(value_type(val)));
  }
  else if (std::isalnum(c)) {
    in.putback(c);
    std::string name = parse_parameter_name(in);
    in>>c;
    if(in && c=='(')
      term_.reset(new Function<T>(in,name));
    else  {
      if (in && !in.eof())
        in.putback(c);
      term_.reset(new Symbol<T>(name));
    }
  }
  else if (c=='(')
    term_.reset(new Block<T>(in));
  else
    boost::throw_exception(std::runtime_error("Illegal term in expression"));
}


template<class T>
Factor<T>::Factor(std::istream& in, bool inv) 
 : super_type(in)
 , is_inverse_(inv)
 , power_(1.)
{
  char c;
  in >> c;
  if (in) {
    if (c=='^') {
      SimpleFactor<T> p(in);
      power_=p;
    }
    else
      in.putback(c);
  }
}

template<class T>
void SimpleFactor<T>::partial_evaluate(const Evaluator<T>& p, bool isarg)
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  Evaluatable<T>* e=term_->partial_evaluate_replace(p,isarg);
  if(e!=term_.get()) term_.reset(e);
}


template<class T>
void Factor<T>::partial_evaluate(const Evaluator<T>& p,bool isarg)
{
  super_type::partial_evaluate(p,isarg);
  power_.partial_evaluate(p,isarg);
}


template<class T>
const SimpleFactor<T>& SimpleFactor<T>::operator=(const SimpleFactor<T>& v)
{
  if (v.term_)
    term_.reset(v.term_->clone());
  else
    term_.reset();
  return *this;
}


template<class T>
bool SimpleFactor<T>::can_evaluate(const Evaluator<T>& p, bool isarg) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  return term_->can_evaluate(p,isarg);
}

template<class T>
bool Factor<T>::can_evaluate(const Evaluator<T>& p, bool isarg) const
{
  return super_type::can_evaluate(p,unit_power() ? isarg : true) && power_.can_evaluate(p,true);
}

template <class T>
typename SimpleFactor<T>::value_type SimpleFactor<T>::value(const Evaluator<T>& p, bool isarg) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  return term_->value(p,isarg);
}

template <class T>
typename Factor<T>::value_type Factor<T>::value(const Evaluator<T>& p, bool isarg) const
{
  value_type val = super_type::value(p,unit_power() ? isarg : true);
  if (is_inverse())
    val = 1./val;
  if (!unit_power())
    val = std::pow(evaluate_helper<T>::real(val),evaluate_helper<T>::real(power_.value(p,true)));
  return val;
}

template<class T>
void SimpleFactor<T>::output(std::ostream& os) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  term_->output(os);
}

template<class T>
void Factor<T>::output(std::ostream& os) const
{
  super_type::output(os);
  if (!unit_power())
    os << "^" << power_;
}


template<class T>
boost::shared_ptr<Factor<T> > Factor<T>::flatten_one_value()
{
  if (unit_power()) {
    boost::shared_ptr<Evaluatable<T> > term=super_type::term_->flatten_one();
    boost::shared_ptr<Factor<T> > val(new Factor<T>(*this));
    val->term_=term;
    return val->term_ ? val : boost::shared_ptr<Factor<T> >();
  }
  else
    return boost::shared_ptr<Factor<T> >();
}

} // end namespace expression
} // end namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace expression {
#endif

template<class T>
inline bool operator==(const alps::expression::Factor<T>& ex1, const alps::expression::Factor<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) ==
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator==(const alps::expression::Factor<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) == s;
}

template<class T>
inline bool operator==(const std::string& s, const alps::expression::Factor<T>& ex)
{
  return ex == s;
}

template<class T>
inline bool operator<(const alps::expression::Factor<T>& ex1, const alps::expression::Factor<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) <
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator<(const alps::expression::Factor<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) < s;
}

template<class T>
inline bool operator<(const std::string& s, const alps::expression::Factor<T>& ex)
{
  return s < boost::lexical_cast<std::string>(ex);
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace expression
} // end namespace alps
#endif

#endif
