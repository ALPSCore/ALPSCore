/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_EXPRESSION2_IMPL_H
#define ALPS_EXPRESSION2_IMPL_H

#include <alps/config.h>

#ifndef ALPS_WITH_NEW_EXPRESSION

#else

#include <alps/expression2.h>

#include <cmath>
#include <stdexcept>
#ifndef BOOST_NO_STRINGSTREAM
# include <sstream>
#else
# include <strstream>
#endif

namespace alps {
namespace expression {

//
// implementation of Evaluator<T>
//

template<class T>
bool Evaluator<T>::can_evaluate(const std::string&) const
{
  return false;
}

template<class T>
bool Evaluator<T>::can_evaluate_function(const std::string& name, const Expression<T>& arg) const
{
  return arg.can_evaluate(*this) &&
         (name=="sqrt" || name=="abs" ||
          name=="sin" || name=="cos" || name=="tan" ||
          name=="log" || name=="exp");
}

template<class T>
typename Evaluator<T>::Direction Evaluator<T>::direction() const
{
  return Evaluator<T>::left_to_right;
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate(const std::string& name) const
{
  return partial_evaluate(name).value();
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate_function(const std::string& name, const Expression<T>& arg) const
{
  return partial_evaluate_function(name,arg).value();
}

template<class T>
Expression<T> Evaluator<T>::partial_evaluate(const std::string& name) const
{
  return Expression<T>(name);
}

template<class T>
Expression<T> Evaluator<T>::partial_evaluate_function(const std::string& name, const Expression<T>& arg) const
{
  if(!arg.can_evaluate(*this)) {
    Expression<T> e(arg);
    e.partial_evaluate(*this);
    return Expression<T>(Function<T>(name,e));
  }
  value_type val=arg.value(*this);
  if (name=="sqrt")
    val = std::sqrt(val);
  else if (name=="abs")
    val = std::abs(val);
  else if (name=="sin")
    val = std::sin(val);
  else if (name=="cos")
    val = std::cos(val);
  else if (name=="tan")
    val = std::tan(val);
  else if (name=="exp")
    val = std::exp(val);
  else if (name=="log")
    val = std::log(val);
  else
    return Expression<T>(Function<T>(name,Expression<T>(val)));
  return Expression<T>(val);
}

//
// implementation of ParameterEvaluator<T>
//

template<class T>
bool ParameterEvaluator<T>::can_evaluate(const std::string& name) const
{
  if (evaluate_helper<T>::can_evaluate_symbol(name)) return true;
  if (!parms_.defined(name) || !parms_[name].valid()) return false;
  Parameters parms(parms_);
  parms[name] = ""; // set illegal to avoid infinite recursion
  return Expression<T>(parms_[name]).can_evaluate(ParameterEvaluator<T>(parms));
}

template<class T>
Expression<T> ParameterEvaluator<T>::partial_evaluate(const std::string& name) const
{
  if (ParameterEvaluator<T>::can_evaluate(name))
    return Expression<T>(ParameterEvaluator<T>::evaluate(name));
  if(!parms_.defined(name))
    return Expression<T>(name);
  Parameters p(parms_);
  p[name]="";
  Expression<T> e(static_cast<std::string>(parms_[name]));
  e.partial_evaluate(ParameterEvaluator<T>(p));
  return e;
}

template<class T>
typename ParameterEvaluator<T>::value_type ParameterEvaluator<T>::evaluate(const std::string& name) const
{
  if (evaluate_helper<T>::can_evaluate_symbol(name))
    return evaluate_helper<T>::evaluate_symbol(name);
  if (parms_[name].template get<std::string>()=="Infinite recursion check" )
    boost::throw_exception(std::runtime_error("Infinite recursion when evaluating " + name));
  Parameters parms(parms_);
  parms[name] = "Infinite recursion check";
  return alps::evaluate<value_type>(parms_[name], ParameterEvaluator<T>(parms));
}

//
// implementation of Evaluatable<T>
//

template<class T>
inline Term<T> Evaluatable<T>::term() const { return Term<T>(); }

//
// implementation of Factor<T>
//

template<class T>
Factor<T>::Factor(value_type x) : term_(new Number<T>(x)), is_inverse_(false) {}

template<class T>
Factor<T>::Factor(const std::string& s)
  : term_(new Symbol<T>(s)), is_inverse_(false) {}

template<class T>
Factor<T>::Factor(std::istream& in, bool inv) : term_(), is_inverse_(inv)
{
  char c;
  in >> c;

  // read value
  if (std::isdigit(c) || c=='.' || c=='+' || c=='-') {
    in.putback(c);
    typename Number<T>::real_type val;
    in >> val;
    term_.reset(new Number<T>(value_type(val)));
  }
  else if (std::isalnum(c)) {
    in.putback(c);
    std::string name = parse_parameter_name(in);
    in>>c;
    if(in && c=='(')
      term_.reset(new Function<T>(in,name));
    else  {
      if (in)
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
void Factor<T>::partial_evaluate(const Evaluator<T>& p)
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  Evaluatable<T>* e=term_->partial_evaluate_replace(p);
  if(e!=term_.get()) term_.reset(e);
}

//
// implementation of Term<T>
//

template<class T>
bool Term<T>::depends_on(const std::string& s) const {
  for (factor_iterator it=factors().first; it!=factors().second; ++it)
    if(it->depends_on(s))
      return true;
  return false;
}

template<class T>
void Term<T>::simplify()
{
  std::vector<Factor<T> > s;
  for (typename std::vector<Factor<T> >::iterator it = terms_.begin();
       it != terms_.end(); ++it) {
    if (it->is_single_term()) {
      Term<T> t = it->term();
      if (t.is_negative()) negate();
      std::copy(t.factors().first, t.factors().second,
                std::back_inserter(s));
    } else {
      s.push_back(*it);
    }
  }
  terms_ = s;
}

template<class T>
Term<T>::Term(std::istream& in, bool negate) : is_negative_(negate)
{
  bool is_inverse=false;
  terms_.push_back(Factor<T>(in,is_inverse));
  while (true) {
    char c;
    in >> c;
    if (!in)
      break;
    switch(c) {
      case '*':
        is_inverse=false;
        break;
      case '/':
        is_inverse=true;
        break;
      default:
        in.putback(c);
        return;
    }
    terms_.push_back(Factor<T>(in,is_inverse));
  }
}

template<class T>
typename Term<T>::value_type Term<T>::value(const Evaluator<T>& p) const
{
  value_type val(1.);
  if (p.direction() == Evaluator<T>::left_to_right)
    for (int i = 0; i < terms_.size(); ++i)
      val *= terms_[i].value(p);
  else
    for (int i = terms_.size()-1; i >= 0; --i)
      val *=terms_[i].value(p);
  return (is_negative() ? -val : val);
}

template<class T>
void Term<T>::partial_evaluate(const Evaluator<T>& p)
{
  if (can_evaluate(p)) {
    (*this) = Term<T>(value(p));
  } else {
    value_type val(1);
    if (p.direction() == Evaluator<T>::left_to_right) {
      for (int i=0; i<terms_.size(); ++i) {
        if (terms_[i].can_evaluate(p)) {
          val *= terms_[i].value(p);
          terms_.erase(terms_.begin()+i);
          --i;
        } else {
          terms_[i].partial_evaluate(p);
        }
      }
    } else {
      for (int i = terms_.size()-1; i >= 0; --i) {
        if (terms_[i].can_evaluate(p)) {
          val *= terms_[i].value(p);
          terms_.erase(terms_.begin()+i);
        } else {
          terms_[i].partial_evaluate(p);
        }
      }
    }
    if (val == value_type(0.)) {
      (*this) = Term<T>(value_type(0.));
    } else {
      if (evaluate_helper<T>::real(val) < 0.) {
        is_negative_=!is_negative_;
        val=-val;
      }
      if (val != value_type(1.))
        terms_.insert(terms_.begin(), Factor<T>(val));
    }
  }
}

template<class T>
bool Term<T>::can_evaluate(const Evaluator<T>& p) const
{
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p);
  return can;
}

template<class T>
void Term<T>::output(std::ostream& os) const
{
  if (terms_.empty()) {
    os << "0";
    return;
  }
  if(is_negative())
    os << " - ";
  terms_[0].output(os);
  for (int i=1;i<terms_.size();++i) {
    os << " " << (terms_[i].is_inverse() ? "/" : "*") << " ";
    terms_[i].output(os);
  }
}

//
// implementation of Expression<T>
//

template<class T>
bool Expression<T>::depends_on(const std::string& s) const {
  for(term_iterator it=terms().first; it!=terms().second; ++it)
    if (it->depends_on(s))
      return true;
  return false;
}

template<class T>
void Expression<T>::simplify()
{
  partial_evaluate();
  for (typename std::vector<Term<T> >::iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    it->simplify();
}

template<class T>
Term<T> Expression<T>::term() const
{
  if (!is_single_term())
    boost::throw_exception(std::logic_error("Called term() for multi-term expression"));
  return terms_[0];
}

template<class T>
void Expression<T>::parse(const std::string& str)
{
#ifndef BOOST_NO_STRINGSTREAM
  std::istringstream in(str);
#else
  std::istrstream in(str.c_str()); // for out-of-the-box g++ 2.95.2
#endif
  parse(in);
}

template<class T>
void Expression<T>::parse(std::istream& is)
{
  terms_.clear();
  bool negate=false;
  char c;
  is >> c;
  if (!is)
    return;
  if (c=='-')
    negate=true;
  else if (c=='+')
    negate=false;
  else
    is.putback(c);
  terms_.push_back(Term<T>(is,negate));
  while(true) {
    is >> c;
    if (!is)
      return;
    if (c=='-')
      negate=true;
    else if (c=='+')
      negate=false;
    else {
      is.putback(c);
      return;
    }
    terms_.push_back(Term<T>(is,negate));
  }
}

//
// implementation of Block<T>
//

template<class T>
Block<T>::Block(std::istream& in) : Expression<T>(in)
{
  char c;
  in >> c;
  if (c != ')' && c != ',')
    boost::throw_exception(std::runtime_error(") or , expected in expression"));
  if (c == ',') {
    // read imaginary part
    Expression<T> ex(in);
    Block<T> bl(ex);
    Term<T> term(bl);
    term *= "I";
    *this += term;
    check_character(in,')',") expected in expression");
  }
}

template<class T>
boost::shared_ptr<Evaluatable<T> > Block<T>::flatten_one()
{
  boost::shared_ptr<Expression<T> > ex = BASE_::flatten_one_expression();
  if (ex)
    return boost::shared_ptr<Evaluatable<T> >(new Block<T>(*ex));
  else
    return boost::shared_ptr<Evaluatable<T> >();
}

//
// implementation of Symbol<T>
//

template<class T>
bool Symbol<T>::depends_on(const std::string& s) const {
  return (name_==s);
}

//
// implementation of Function<T>
//

template<class T>
Function<T>::Function(std::istream& in,const std::string& name)
  :  name_(name), arg_(in)
{
  check_character(in,')',") expected after function call");
}

template<class T>
bool Function<T>::depends_on(const std::string& s) const {
  if (name_==s) return true;
  return arg_.depends_on(s);
}

template<class T>
boost::shared_ptr<Evaluatable<T> > Function<T>::flatten_one()
{
  arg_.flatten();
  return boost::shared_ptr<Expression<T> >();
}

//
// implementation of Number<T>
//

template<class T>
typename Expression<T>::value_type Expression<T>::value(const Evaluator<T>& p) const
{
  if (terms_.size()==0)
    return value_type(0.);
  value_type val=terms_[0].value(p);
  for (int i=1;i<terms_.size();++i)
    val += terms_[i].value(p);
  return val;
}

template<class T>
bool Expression<T>::can_evaluate(const Evaluator<T>& p) const
{
  if (terms_.size()==0)
    return true;
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p);
  return can;
}

template<class T>
void Expression<T>::partial_evaluate(const Evaluator<T>& p)
{
  if (can_evaluate(p))
    (*this) = Expression<T>(value(p));
  else {
    value_type val(0);
    for (int i=0; i<terms_.size(); ++i) {
      if (terms_[i].can_evaluate(p)) {
        val += terms_[i].value(p);
        terms_.erase(terms_.begin()+i);
        --i;
      } else {
        terms_[i].partial_evaluate(p);
      }
    }
    if (val != value_type(0.)) terms_.insert(terms_.begin(), Term<T>(val));
  }
}

template<class T>
void Expression<T>::output(std::ostream& os) const
{
  if (terms_.size()==0)
    os <<"0";
  else {
    terms_[0].output(os);
    for (int i=1;i<terms_.size();++i) {
      if(!terms_[i].is_negative())
        os << " + ";
      terms_[i].output(os);
    }
  }
}

template<class T>
const Factor<T>& Factor<T>::operator=(const Factor<T>& v)
{
  if (v.term_) {
    term_.reset(v.term_->clone());
    is_inverse_=v.is_inverse_;
  }
  else {
    is_inverse_=false;
    term_.reset();
  }
  return *this;
}


template<class T>
bool Factor<T>::can_evaluate(const Evaluator<T>& p) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  return term_->can_evaluate(p);
}

template<class T>
void Factor<T>::output(std::ostream& os) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  term_->output(os);
}

template<class T>
void Block<T>::output(std::ostream& os) const
{
  os << "(";
  BASE_::output(os);
  os << ")";
}

template<class T>
Evaluatable<T>* Block<T>::partial_evaluate_replace(const Evaluator<T>& p)
{
  partial_evaluate(p);
  return this;
}

template<class T>
typename Number<T>::value_type Number<T>::value(const Evaluator<T>&) const
{
  return val_;
}

template<class T>
void Number<T>::output(std::ostream& os) const
{
  if (evaluate_helper<T>::imag(val_) == 0)
    os << evaluate_helper<T>::real(val_);
  else
    os << val_;
}

template<class T>
typename Symbol<T>::value_type Symbol<T>::value(const Evaluator<T>& eval) const
{
   return eval.evaluate(name_);
}

template<class T>
Evaluatable<T>* Symbol<T>::partial_evaluate_replace(const Evaluator<T>& p)
{
  Expression<T> e(p.partial_evaluate(name_));
  if (e==name_)
    return this;
  else
    return new Block<T>(p.partial_evaluate(name_));
}

template<class T>
Evaluatable<T>* Function<T>::partial_evaluate_replace(const Evaluator<T>& p)
{
  if (can_evaluate(p))
    return new Expression<T>(value(p));
  else {
    arg_.partial_evaluate(p);
    return this;
  }
}

template<class T>
typename Function<T>::value_type Function<T>::value(const Evaluator<T>& p) const
{
  return p.evaluate_function(name_,arg_);
}

template<class T>
bool Function<T>::can_evaluate(const Evaluator<T>& p) const
{
  return p.can_evaluate_function(name_,arg_);
}

template<class T>
void Function<T>::output(std::ostream& os) const
{
  os << name_ << "(" << arg_ << ")";
}

// Parameters evaluate(const Parameters& in)
// {
//   Parameters out;
//   ParameterEvaluator eval(in);
//   for (Parameters::const_iterator p = in.begin(); p != in.end(); ++p) {
//     const std::string name = p->key();
//     const std::string value = static_cast<std::string>(p->value());
//     Expression e(value);
//     if (e.can_evaluate(eval)) {
//       // if value can be evaluated, then replace it by the evaluated value
//       out[name] = e.value(eval);
//     } else {
//       // if value cannot be evaluated, then it remains untouched
//       out[name] = value;
//     }
//   }
//   return out;
// }

template<class T>
void Expression<T>::flatten()
{
  int i=0;
  while (i<terms_.size()) {
    boost::shared_ptr<Term<T> > term = terms_[i].flatten_one_term();
    if (term)
      terms_.insert(terms_.begin()+i,*term);
    else
      ++i;
  }
}

template<class T>
boost::shared_ptr<Term<T> > Term<T>::flatten_one_term()
{
  for (int i=0;i<terms_.size();++i)
    if (!terms_[i].is_inverse()) {
      boost::shared_ptr<Factor<T> > val = terms_[i].flatten_one_value();
      if (val) {
        boost::shared_ptr<Term<T> > term(new Term<T>(*this));
        term->terms_[i]=*val;
        return term;
      }
  }
  return boost::shared_ptr<Term>();
}

template<class T>
boost::shared_ptr<Factor<T> > Factor<T>::flatten_one_value()
{
  boost::shared_ptr<Evaluatable<T> > term=term_->flatten_one();
  boost::shared_ptr<Factor<T> > val(new Factor<T>(*this));
  val->term_=term;
  return val->term_ ? val : boost::shared_ptr<Factor<T> >();
}

template<class T>
boost::shared_ptr<Expression<T> > Expression<T>::flatten_one_expression()
{
  flatten();
  if (terms_.size()>1) {
    boost::shared_ptr<Expression<T> > term(new Expression<T>());
    term->terms_.push_back(terms_[0]);
    terms_.erase(terms_.begin());
    return term;
  }
  else
    return boost::shared_ptr<Expression<T> >();
}

} // end namespace expression
} // end namespace alps

#endif // ! ALPS_WITH_NEW_EXPRESSION

#endif // ! ALPS_EXPRESSION2_IMPL_H
