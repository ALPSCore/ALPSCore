/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_EXPRESSION_IMPL_H
#define ALPS_EXPRESSION_IMPL_H

#include <alps/config.h>
#include <alps/expression.h>
#include <alps/vectorio.h>

#include <cmath>
#include <stdexcept>

namespace alps {

namespace expression {

//
// implementation of Evaluator<T>
//

template<class T>
bool Evaluator<T>::can_evaluate(const std::string&, bool) const
{
  return false;
}

template<class T>
bool Evaluator<T>::can_evaluate_function(const std::string& name, const Expression<T>& arg, bool) const
{
  return arg.can_evaluate(*this,true) &&
         (name=="sqrt" || name=="abs" ||
          name=="sin" || name=="cos" || name=="tan" ||
          name=="asin" || name=="acos" || name=="atan" ||
          name=="log" || name=="exp" || name=="integer_random");
}


template<class T>
bool Evaluator<T>::can_evaluate_expressions(const std::vector<Expression<T> >& arg, bool f) const
{
  bool can=true;
  for (typename std::vector<Expression<T> >::const_iterator it=arg.begin();it!=arg.end();++it)
    can = can && it->can_evaluate(*this,f);
  return can;
}

template<class T>
void Evaluator<T>::partial_evaluate_expressions(std::vector<Expression<T> >& arg, bool f) const
{
  for (typename std::vector<Expression<T> >::iterator it=arg.begin();it!=arg.end();++it) {
   it->partial_evaluate(*this,f);
   it->simplify();
  }
}


template<class T>
bool Evaluator<T>::can_evaluate_function(const std::string& name, const std::vector<Expression<T> >& arg, bool f) const
{
  return can_evaluate_expressions(arg,true) &&
       ((arg.size()==0 && (name == "random" || name=="gaussian_random" || name == "normal_random")) ||
        (arg.size()==1 && can_evaluate_function(name,arg[0],f)) || 
        (arg.size()==2 && (name=="gaussian_random" || name=="atan2")));
}


template<class T>
typename Evaluator<T>::Direction Evaluator<T>::direction() const
{
  return Evaluator<T>::left_to_right;
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate(const std::string& name,bool isarg) const
{
  return partial_evaluate(name,isarg).value();
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate_function(const std::string& name, const Expression<T>& arg,bool isarg) const
{
  return partial_evaluate_function(name,arg,isarg).value();
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate_function(const std::string& name, const std::vector<Expression<T> >& arg,bool isarg) const
{
  return partial_evaluate_function(name,arg,isarg).value();
}

template<class T>
Expression<T> Evaluator<T>::partial_evaluate(const std::string& name,bool) const
{
  return Expression<T>(name);
}


template<class T>
Expression<T> Evaluator<T>::partial_evaluate_function(const std::string& name, const Expression<T>& arg,bool) const
{
  if(!arg.can_evaluate(*this,true)) {
    Expression<T> e(arg);
    e.partial_evaluate(*this,true);
    return Expression<T>(Function<T>(name,e));
  }
  value_type val=arg.value(*this,true);
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
  else if (name=="asin")
    val = std::asin(evaluate_helper<T>::real(val));
  else if (name=="acos")
    val = std::acos(evaluate_helper<T>::real(val));
  else if (name=="atan")
    val = std::atan(evaluate_helper<T>::real(val));
  else if (name=="exp")
    val = std::exp(val);
  else if (name=="log")
    val = std::log(val);
  else if (name=="integer_random")
    val=static_cast<int>(evaluate_helper<T>::real(val)*Disorder::random());
  else
    return Expression<T>(Function<T>(name,Expression<T>(val)));
  return Expression<T>(val);
}

template<class T>
Expression<T> Evaluator<T>::partial_evaluate_function(const std::string& name, const std::vector<Expression<T> >& args,bool isarg) const
{
  if (args.size()==1)
    return partial_evaluate_function(name,args[0],isarg);
    
  std::vector<Expression<T> > evaluated;
  bool could_evaluate = true;
  for (typename std::vector<Expression<T> >::const_iterator it = args.begin(); it !=args.end();++it) {
    evaluated.push_back(*it);
    could_evaluate = could_evaluate && it->can_evaluate(*this,true);
    evaluated.rbegin()->partial_evaluate(*this,true);
  }
  if (evaluated.size()==2 && could_evaluate) {
    double arg1=evaluate_helper<T>::real(evaluated[0].value());
    double arg2=evaluate_helper<T>::real(evaluated[1].value());
    if (name=="atan2")
      return Expression<T>(static_cast<T>(std::atan2(arg1,arg2)));
    else if (name=="gaussian_random" || name=="normal_random")
      return Expression<T>(arg1+arg2*Disorder::gaussian_random());
  }
  else if (evaluated.size()==0) {
    if (name=="random")
      return Expression<T>(Disorder::random());
    else if (name=="gaussian_random" || name=="normal_random")
      return Expression<T>(Disorder::gaussian_random());
  }
  return Expression<T>(Function<T>(name,evaluated));
}


//
// implementation of ParameterEvaluator<T>
//

template<class T>
bool ParameterEvaluator<T>::can_evaluate(const std::string& name, bool isarg) const
{
  if (evaluate_helper<T>::can_evaluate_symbol(name,isarg)) return true;
  if (!parms_.defined(name) || !parms_[name].valid()) return false;
  Parameters parms(parms_);
  parms[name] = ""; // set illegal to avoid infinite recursion
  return Expression<T>(parms_[name]).can_evaluate(ParameterEvaluator<T>(parms),isarg);
}

template<class T>
Expression<T> ParameterEvaluator<T>::partial_evaluate(const std::string& name, bool isarg) const
{
  Expression<T> e;
  if (ParameterEvaluator<T>::can_evaluate(name,isarg))
    e=ParameterEvaluator<T>::evaluate(name,isarg);
  else if(!parms_.defined(name))
    e=Expression<T>(name);
  else {
    Parameters p(parms_);
    p[name]="";
    e=Expression<T>(static_cast<std::string>(parms_[name]));
    e.partial_evaluate(ParameterEvaluator<T>(p),isarg);
  }
  return e;
}

template<class T>
typename ParameterEvaluator<T>::value_type ParameterEvaluator<T>::evaluate(const std::string& name, bool isarg) const
{
  if (evaluate_helper<T>::can_evaluate_symbol(name,isarg))
    return evaluate_helper<T>::evaluate_symbol(name,isarg);
  if (parms_[name].template get<std::string>()=="Infinite recursion check" )
    boost::throw_exception(std::runtime_error("Infinite recursion when evaluating " + name));
  Parameters parms(parms_);
  parms[name] = "Infinite recursion check";
  return alps::evaluate<value_type>(parms_[name], ParameterEvaluator<T>(parms), isarg);
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
SimpleFactor<T>::SimpleFactor(std::istream& in) : term_()
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
Factor<T>::Factor(std::istream& in, bool inv) : super_type(in), is_inverse_(inv), power_(1.)
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
  partial_evaluate();
}

template<class T>
void Term<T>::remove_spurious_parentheses()
{
  std::vector<Factor<T> > s;
  for (typename std::vector<Factor<T> >::iterator it = terms_.begin();
       it != terms_.end(); ++it) {
    if (it->is_single_term()) {
      Term<T> t = it->term();
      if (t.is_negative()) negate();
      std::copy(t.factors().first, t.factors().second,
                std::back_inserter(s));
    } else
      s.push_back(*it);
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
typename Term<T>::value_type Term<T>::value(const Evaluator<T>& p, bool isarg) const
{
  value_type val(1.);
  if (p.direction() == Evaluator<T>::left_to_right)  {
    for (int i = 0; i < terms_.size() && is_nonzero(val); ++i)
      val *= terms_[i].value(p,isarg);
}
  else {
    for (int i = terms_.size()-1; i >= 0 && is_nonzero(val); --i) {
      value_type tmp=terms_[i].value(p,isarg);
      val *=tmp;
    }
  }
  if (is_negative() && is_nonzero(val))
    val = val*(-1.);
  return val;
}

template<class T>
void Term<T>::partial_evaluate(const Evaluator<T>& p, bool isarg)
{
  if (can_evaluate(p,isarg)) {
    (*this) = Term<T>(value(p,isarg));
  } else {
    value_type val(1);
    if (p.direction() == Evaluator<T>::left_to_right) {
      for (int i=0; i<terms_.size(); ++i) {
        if (terms_[i].can_evaluate(p,isarg)) {
          val *= terms_[i].value(p,isarg);
          if (is_zero(val))
            break;
          terms_.erase(terms_.begin()+i);
          --i;
        } else {
          terms_[i].partial_evaluate(p,isarg);
        }
      }
    } else {
      for (int i = terms_.size()-1; i >= 0; --i) {
        if (terms_[i].can_evaluate(p,isarg)) {
          val *= terms_[i].value(p,isarg);
          if (is_zero(val))
            break;
          terms_.erase(terms_.begin()+i);
        } else
          terms_[i].partial_evaluate(p,isarg);
      }
    }
    if (is_zero(val))
      (*this) = Term<T>(value_type(0.));
    else {
      if (evaluate_helper<T>::real(val) < 0.) {
        is_negative_=!is_negative_;
        val=-val;
      }
      if (val != value_type(1.))
        terms_.insert(terms_.begin(), Factor<T>(val));
    }
  }
  remove_spurious_parentheses();
}

template<class T>
bool Term<T>::can_evaluate(const Evaluator<T>& p, bool isarg) const
{
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p,isarg);
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

template<class T>
Term<T>::Term(const std::pair<T,Term<T> >& t)
 :  is_negative_(false),terms_(t.second.terms_)
{
  terms_.insert(terms_.begin(), Factor<T>(t.first));
  partial_evaluate();
}

template<class T>
std::pair<T,Term<T> > Term<T>::split() const
{
  Term<T> t(*this);
  t.partial_evaluate();
  T val=1.;
  if (t.terms_.empty())
    val=0.;
  else
    if (t.terms_[0].can_evaluate()) {
    val=t.terms_[0].value();
    t.terms_.erase(t.terms_.begin());
  }
  if (t.is_negative_) 
    val=-val;
  t.is_negative_=false;
  return std::make_pair(val,t);
}

template <class T>
struct term_less {
  bool operator()(const Term<T>& x, const Term<T>& y) {
    return x.split().second < y.split().second;
  }
};


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
void Expression<T>::remove_spurious_parentheses()
{
  for (typename std::vector<Term<T> >::iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    it->remove_spurious_parentheses();
}

template<class T>
void Expression<T>::simplify()
{
  partial_evaluate();
  for (typename std::vector<Term<T> >::iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    it->simplify();
  sort();
  partial_evaluate();
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

template <class T>
void Expression<T>::sort()
{
  partial_evaluate();
  std::sort(terms_.begin(),terms_.end(),term_less<T>());
  typename std::vector<Term<T> >::iterator prev,it;
  prev=terms_.begin();
  if (prev==terms_.end())
    return;
  it=prev;
  ++it;
  bool added=false;
  std::pair<T,Term<T> > prev_term=prev->split();
  while (it !=terms_.end()) {
    std::pair<T,Term<T> > current_term=it->split();
    
    if (prev_term.second==current_term.second) {
      prev_term.first += current_term.first;
      terms_.erase(it);
      added=true;
      *prev=Term<T>(prev_term);
    }
    else {
      if (added && is_zero(prev_term.first))
        terms_.erase(prev);
      else {
        prev=it;
        ++it;
      }
      prev_term=current_term;
    }
    added=false;
  }
  if (added && is_zero(prev_term.first))
    terms_.erase(prev);
}

template<class T>
typename Expression<T>::value_type Expression<T>::value(const Evaluator<T>& p, bool isarg) const
{
  if (terms_.size()==0)
    return value_type(0.);
  value_type val=terms_[0].value(p);
  for (int i=1;i<terms_.size();++i)
    val += terms_[i].value(p,isarg);
  return val;
}

template<class T>
bool Expression<T>::can_evaluate(const Evaluator<T>& p, bool isarg) const
{
  if (terms_.size()==0)
    return true;
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p,isarg);
  return can;
}

template<class T>
void Expression<T>::partial_evaluate(const Evaluator<T>& p, bool isarg)
{
  if (can_evaluate(p,isarg))
    (*this) = Expression<T>(value(p,isarg));
  else {
    value_type val(0);
    for (int i=0; i<terms_.size(); ++i) {
      if (terms_[i].can_evaluate(p,isarg)) {
        val += terms_[i].value(p,isarg);
        terms_.erase(terms_.begin()+i);
        --i;
      } else {
        terms_[i].partial_evaluate(p,isarg);
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
  :  name_(name), args_()
{
  char c;
  in >> c;
  if (c!=')') {
    in.putback(c);
    do {
      args_.push_back(Expression<T>(in));
      in >> c;
    } while (c==',');
    if (c!=')')
      boost::throw_exception(std::runtime_error(std::string("received ") + c + " instead of ) at end of function argument list"));
  }
}

template<class T>
bool Function<T>::depends_on(const std::string& s) const {
  if (name_==s) return true;
  for (typename std::vector<Expression<T> >::const_iterator it=args_.begin();it != args_.end();++it)
    if (it->depends_on(s))
      return true;
  return false;
}

template<class T>
boost::shared_ptr<Evaluatable<T> > Function<T>::flatten_one()
{
  for (typename std::vector<Expression<T> >::iterator it=args_.begin();it != args_.end();++it)
    it->flatten();
  return boost::shared_ptr<Expression<T> >();
}

//
// implementation of Number<T>
//


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
void Block<T>::output(std::ostream& os) const
{
  os << "(";
  BASE_::output(os);
  os << ")";
}

template<class T>
Evaluatable<T>* Block<T>::partial_evaluate_replace(const Evaluator<T>& p, bool isarg)
{
  partial_evaluate(p,isarg);
  return this;
}

template<class T>
typename Number<T>::value_type Number<T>::value(const Evaluator<T>&, bool) const
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
typename Symbol<T>::value_type Symbol<T>::value(const Evaluator<T>& eval, bool isarg) const
{
  if (!eval.can_evaluate(name_,isarg))
    boost::throw_exception(std::runtime_error("Cannot evaluate " + name_ ));
  return eval.evaluate(name_,isarg);
}

template<class T>
Evaluatable<T>* Symbol<T>::partial_evaluate_replace(const Evaluator<T>& p, bool isarg)
{
  Expression<T> e(p.partial_evaluate(name_,isarg));
  if (e==name_)
    return this;
  else
    return new Block<T>(p.partial_evaluate(name_,isarg));
}

template<class T>
Evaluatable<T>* Function<T>::partial_evaluate_replace(const Evaluator<T>& p, bool isarg)
{
  p.partial_evaluate_expressions(args_,true);
  return new Block<T>(p.partial_evaluate_function(name_,args_,isarg));
}

template<class T>
typename Function<T>::value_type Function<T>::value(const Evaluator<T>& p, bool isarg) const
{
  value_type val=p.evaluate_function(name_,args_,isarg);
  return val;
}

template<class T>
bool Function<T>::can_evaluate(const Evaluator<T>& p, bool isarg) const
{
  return p.can_evaluate_function(name_,args_,isarg);
}

template<class T>
void Function<T>::output(std::ostream& os) const
{
  os << name_ << "(" << vector_writer(args_,", ") << ")";
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
  if (unit_power()) {
    boost::shared_ptr<Evaluatable<T> > term=super_type::term_->flatten_one();
    boost::shared_ptr<Factor<T> > val(new Factor<T>(*this));
    val->term_=term;
    return val->term_ ? val : boost::shared_ptr<Factor<T> >();
  }
  else
    return boost::shared_ptr<Factor<T> >();
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

#endif // ! ALPS_EXPRESSION_IMPL_H
