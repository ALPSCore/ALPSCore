/***************************************************************************
* ALPS library
*
* alps/expression.C   A class to evaluate expressions
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/expression.h>
#include <alps/expression_impl.h>

#include <alps/cctype.h>
#include <alps/parameters.h>
#include <alps/parser/parser.h>

#include <boost/throw_exception.hpp>
#include <cmath>
#include <stdexcept>
#ifndef BOOST_NO_STRINGSTREAM
# include <sstream>
#else
# include <strstream>
#endif

namespace alps {

namespace detail {

Evaluatable::operator double() const {
  if (!can_evaluate()) {
    std::cerr << "Was trying to evaluate " << *this << "\n";
    boost::throw_exception(std::runtime_error("Cannot evaluate expression"));
  }
  return value();
}

double Evaluatable::value() const
{
  return value(Evaluator());
}

boost::shared_ptr<Evaluatable> Evaluatable::flatten_one() 
{
  return boost::shared_ptr<Evaluatable>();
}

Evaluatable* Evaluatable::partial_evaluate_replace(const Evaluator& )
{
  return this;
}


bool Evaluatable::can_evaluate() const
{
  return can_evaluate(Evaluator());
}

} // namespace detail

Expression::Expression(const std::string& expression )
{
#ifndef BOOST_NO_STRINGSTREAM
  std::istringstream str(expression);
#else
  std::istrstream str(expression.c_str()); // for out-of-the-box g++ 2.95.2
#endif
  (*this) = Expression(str);
}

Expression::Expression(std::istream& in)
{
  bool negate=false;
  char c;
  in >> c;
  if (!in)
    return;
  if (c=='-') 
    negate=true;
  else if (c=='+')
    negate=false;
  else
    in.putback(c);
  terms_.push_back(Term(in,negate));
  while(true) {
    in >> c;
    if (!in)
      return;
    if (c=='-') 
      negate=true;
    else if (c=='+')
      negate=false;
    else {
      in.putback(c);
      return;
    }
    terms_.push_back(Term(in,negate));
  }
}

const Expression& Expression::operator+=(const Expression& e)
{
  std::copy(e.terms_.begin(),e.terms_.end(),std::back_inserter(terms_));
  partial_evaluate();
  return *this;
}

double Expression::value(const Evaluator& p) const
{
  if (terms_.size()==0)
    boost::throw_exception(std::runtime_error("Empty expression"));
  double val=terms_[0].value(p);
  for (int i=1;i<terms_.size();++i)
    val +=terms_[i].value(p);
  return val;
}

bool Expression::can_evaluate(const Evaluator& p) const
{
  if (terms_.size()==0)
    return false;
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p);
  return can;
}

void Expression::partial_evaluate(const Evaluator& p)
{
  if (can_evaluate(p))
    (*this) = Expression(value(p));
  else {
    double val=0.;
    for (int i=0;i<terms_.size();++i)
      if (terms_[i].can_evaluate(p)) {
        val += terms_[i].value(p);
        terms_.erase(terms_.begin()+i);
        --i;
      }
      else
        terms_[i].partial_evaluate(p);
    if (val!=0)
      terms_.insert(terms_.begin(),Term(val));
  }
}

void Expression::output(std::ostream& out) const
{
  if (terms_.size()==0)
    out <<"0";
  else {
    terms_[0].output(out);
    for (int i=1;i<terms_.size();++i) {
      if(!terms_[i].is_negative())
        out << " + ";
      terms_[i].output(out);
    }
  }
}

detail::Evaluatable* Expression::clone() const
{
  return new Expression(*this);
}

Term::Term(std::istream& in, bool negate)
 : is_negative_(negate)
{
  bool is_inverse=false;
  terms_.push_back(detail::Value(in,is_inverse));
  while (true) {
    char c;
    in >> c;
    if (!in)
      break;
    switch(c) {
      case '*':
        is_inverse=false;;
        break;
      case '/':
        is_inverse=true;
        break;
      default:
        in.putback(c);
        return;
    }
    terms_.push_back(detail::Value(in,is_inverse));
  }
}

double Term::value(const Evaluator& p) const
{
  double val=1.;
  if (p.direction()==Evaluator::left_to_right)
    for (int i=0;i<terms_.size();++i)
      val *=terms_[i].value(p);
  else
    for (int i=terms_.size()-1;i>=0;--i)
      val *=terms_[i].value(p);
  return (is_negative() ? -val : val);
}

void Term::partial_evaluate(const Evaluator& p)
{
  if (can_evaluate(p))
    (*this) = Term(value(p));
  else {
    double val=1.;
    if (p.direction()==Evaluator::left_to_right) {
      for (int i=0;i<terms_.size();++i)
        if (terms_[i].can_evaluate(p)) {
          val *= terms_[i].value(p);
          terms_.erase(terms_.begin()+i);
          --i;
        }
        else
          terms_[i].partial_evaluate(p);
    }
    else {
      for (int i=terms_.size()-1;i>=0;--i) 
        if (terms_[i].can_evaluate(p)) {
          val *= terms_[i].value(p);
          terms_.erase(terms_.begin()+i);
        }
        else
          terms_[i].partial_evaluate(p);
    }
    if(val==0.)
      (*this)=Term(0.);
    else {
      if (val<0.) {
        is_negative_=!is_negative_;
	val=-val;
      }
      if (val!=1.)
        terms_.insert(terms_.begin(),detail::Value(val));
    }
  }
}

bool Term::can_evaluate(const Evaluator& p) const
{
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p);
  return can;
}

void Term::output(std::ostream& out) const
{
  if(is_negative())
    out << " - ";
  terms_[0].output(out);
  for (int i=1;i<terms_.size();++i) {
    out << " " << (terms_[i].is_inverse() ? "/" : "*") << " ";
    terms_[i].output(out);
  }
}

detail::Evaluatable* Term::clone() const
{
  return new Term(*this);
}


detail::Value::Value(std::istream& in, bool inv)
 : term_(), is_inverse_(inv)
{
  char c;
  in >> c;

  // read value
  if (std::isdigit(c) || c=='.' || c=='+' || c=='-') {
    in.putback(c);
    double val;
    in>>val;
    term_.reset(new Number(val));
  }
  else if (std::isalnum(c)) {
    in.putback(c);
    std::string name=parse_identifier(in);
    in>>c;
    if(in && c=='(')
      term_.reset(new Function(in,name));
    else  {
      if (in)
        in.putback(c);
      term_.reset(new Symbol(name));
    }
  }
  else if (c=='(')
    term_.reset(new Block(in));
  else
    boost::throw_exception(std::runtime_error("Illegal term in expression"));
}


detail::Value::Value(const detail::Value& v)
 : Evaluatable(v), term_(), is_inverse_(v.is_inverse_)
{
  if (v.term_)
    term_.reset(v.term_->clone());
}

const detail::Value& detail::Value::operator=(const detail::Value& v)
{
  if (v.term_)
    term_.reset(v.term_->clone());
  else
    term_.reset();
  return *this;
}

detail::Evaluatable* detail::Value::clone() const
{
  return new Value(*this);
}

detail::Value::Value(double x) 
 : term_(new Number(x)), is_inverse_(false) 
{
}

double detail::Value::value(const Evaluator& p) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  return is_inverse() ? 1./term_->value(p) : term_->value(p);
}

bool detail::Value::can_evaluate(const Evaluator& p) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  return term_->can_evaluate(p);
}

void detail::Value::partial_evaluate(const Evaluator& p)
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));
  Evaluatable* e=term_->partial_evaluate_replace(p);
  if(e!=term_.get() )
    term_.reset(e);
}

void detail::Value::output(std::ostream& out) const
{
  if (!term_)
    boost::throw_exception(std::runtime_error("Empty value in expression"));  
  term_->output(out);
}

detail::Block::Block(std::istream& in)
 : Expression(in)
{
  check_character(in,')',") expected in expression");
}

void detail::Block::output(std::ostream& out) const
{
  out << "(";
  Expression::output(out);
  out << ")";
}

detail::Evaluatable* detail::Block::clone() const
{
  return new Block(*this);
}

detail::Evaluatable* detail::Block::partial_evaluate_replace(const Evaluator& p)
{
  partial_evaluate(p);
  return this;
}

bool detail::Number::can_evaluate(const Evaluator&) const
{
  return true;
}

double detail::Number::value(const Evaluator&) const
{
  return val_;
}

void detail::Number::output(std::ostream& out) const
{
  out << val_;
}

detail::Evaluatable* detail::Number::clone() const
{
  return new Number(*this);
}

double detail::Symbol::value(const Evaluator& eval) const
{
   return eval.evaluate(name_);
}

detail::Evaluatable* detail::Symbol::partial_evaluate_replace(const Evaluator& p)
{
  Expression e(p.partial_evaluate(name_));
  if (e==name_)
    return this;
  else
    return new detail::Block(p.partial_evaluate(name_));
}

bool detail::Symbol::can_evaluate(const Evaluator& eval) const
{
  return eval.can_evaluate(name_);
}

void detail::Symbol::output(std::ostream& out) const
{
  out << name_;
}

detail::Evaluatable* detail::Symbol::clone() const
{
  return new Symbol(*this);
}

detail::Function::Function(std::istream& in,const std::string& name)
 :  name_(name), arg_(in)
{
  check_character(in,')',") expected after function call");
}

detail::Evaluatable* detail::Function::partial_evaluate_replace(const Evaluator& p)
{
  if (can_evaluate(p))
    return new Expression(value(p));
  else {
    arg_.partial_evaluate(p);
    return this;
  }
}

double detail::Function::value(const Evaluator& p) const
{
  return p.evaluate_function(name_,arg_);
}

bool detail::Function::can_evaluate(const Evaluator& p) const
{
  return p.can_evaluate_function(name_,arg_);
}

void detail::Function::output(std::ostream& out) const
{
  out << name_ << "(" << arg_ << ")";
}

detail::Evaluatable* detail::Function::clone() const
{
  return new Function(*this);
}


Parameters evaluate(const Parameters& in) 
{
  Parameters out;
  ParameterEvaluator eval(in);
  for (Parameters::const_iterator p = in.begin(); p != in.end(); ++p) {
    const std::string name = p->key();
    const std::string value = static_cast<std::string>(p->value());
    Expression e(value);
    if (e.can_evaluate(eval)) {
      // if value can be evaluated, then replace it by the evaluated value
      out[name] = e.value(eval);
    } else {
      // if value cannot be evaluated, then it remains untouched
      out[name] = value;
    }
  }
  return out;
}

void Expression::flatten() 
{
  int i=0;
  while (i<terms_.size()) {
    boost::shared_ptr<Term> term = terms_[i].flatten_one_term();
    if (term)
      terms_.insert(terms_.begin()+i,*term);
    else
      ++i;
  }
}

boost::shared_ptr<Term> Term::flatten_one_term()
{
  for (int i=0;i<terms_.size();++i) 
    if (!terms_[i].is_inverse()) {
      boost::shared_ptr<detail::Value> val = terms_[i].flatten_one_value();
      if (val) {
        boost::shared_ptr<Term> term(new Term(*this));
        term->terms_[i]=*val;
        return term;
      }
  }
  return boost::shared_ptr<Term>();
}

boost::shared_ptr<detail::Value> detail::Value::flatten_one_value()
{
  boost::shared_ptr<detail::Evaluatable> term=term_->flatten_one();
  boost::shared_ptr<Value> val(new Value(*this));
  val->term_=term;
  return val->term_ ? val : boost::shared_ptr<detail::Value>();
}

boost::shared_ptr<Expression> Expression::flatten_one_expression()
{
  flatten();
  if (terms_.size()>1) {
    boost::shared_ptr<Expression> term(new Expression());
    term->terms_.push_back(terms_[0]);
    terms_.erase(terms_.begin());
    return term;
  }
  else 
    return boost::shared_ptr<Expression>();
}

boost::shared_ptr<detail::Evaluatable> detail::Function::flatten_one()
{
  arg_.flatten();
  return boost::shared_ptr<Expression>();
}

boost::shared_ptr<detail::Evaluatable> detail::Block::flatten_one()
{
  boost::shared_ptr<Expression> ex = flatten_one_expression();
  if (ex)
    return boost::shared_ptr<Evaluatable>(new Block(*ex));
  else
    return boost::shared_ptr<Evaluatable>();
}

} // namespace alps
