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

double Evaluatable::value() const
{
  return value(Parameters());
}

boost::shared_ptr<Evaluatable> Evaluatable::flatten_one() 
{
  return boost::shared_ptr<Evaluatable>();
}

bool Evaluatable::can_evaluate() const
{
  return can_evaluate(Parameters());
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

double Expression::value(const Parameters& p) const
{
  if (terms_.size()==0)
    boost::throw_exception(std::runtime_error("Empty expression"));
  double val=terms_[0].value(p);
  for (int i=1;i<terms_.size();++i)
    val +=terms_[i].value(p);
  return val;
}

bool Expression::can_evaluate(const Parameters& p) const
{
  if (terms_.size()==0)
    return false;
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p);
  return can;
}

void Expression::output(std::ostream& out) const
{
  if (terms_.size()==0)
    return;
  if (terms_[0].is_negative())
    out << "-";
  terms_[0].output(out);
  for (int i=1;i<terms_.size();++i) {
    out << (terms_[i].is_negative() ? " - " : " + ");
    terms_[i].output(out);
  }
}

detail::Evaluatable* Expression::clone() const
{
  return new Expression(*this);
}

Term::Term(std::istream& in, bool negate)
 : is_negative_(negate)
{
  terms_.push_back(detail::Value(in));
  while (true) {
    char c;
    in >> c;
    if (!in)
      break;
    switch(c) {
      case '*':
        ops_.push_back(MULTIPLY);
        break;
      case '/':
        ops_.push_back(DIVIDE);
        break;
      default:
        in.putback(c);
        return;
    }
    terms_.push_back(detail::Value(in));
  }
}

double Term::value(const Parameters& p) const
{
  double val=terms_[0].value(p);
  for (int i=0;i<ops_.size();++i)
    if (ops_[i]==MULTIPLY)
      val *=terms_[i+1].value(p);
    else 
      val /=terms_[i+1].value(p);
  return (is_negative() ? -val : val);
}

bool Term::can_evaluate(const Parameters& p) const
{
  bool can=true;
  for (int i=0;i<terms_.size();++i)
    can = can && terms_[i].can_evaluate(p);
  return can;
}

void Term::output(std::ostream& out) const
{
  terms_[0].output(out);
  for (int i=0;i<ops_.size();++i)
  {
    out << " " << (ops_[i]==MULTIPLY ? "*" : "/") << " ";
    terms_[i+1].output(out);
  }
}

detail::Evaluatable* Term::clone() const
{
  return new Term(*this);
}

detail::Value::Value(std::istream& in)
 : term_(0)
{
  char c;
  in >> c;

  // read value
  if (std::isdigit(c) || c=='.' || c=='+' || c=='-') {
    in.putback(c);
    double val;
    in>>val;
    term_=new Number(val);
  }
  else if (std::isalnum(c)) {
    in.putback(c);
    std::string name=parse_identifier(in);
    in>>c;
    if(in && c=='(')
      term_=new Function(in,name);
    else  {
      if (in)
        in.putback(c);
      term_=new Symbol(name);
    }
  }
  else if (c=='(')
    term_=new Block(in);
  else
    boost::throw_exception(std::runtime_error("Illegal term in expression"));
}

detail::Value::~Value()
{
  if (term_)
    delete term_;
}

detail::Value::Value(const detail::Value& v)
 : Evaluatable(v), term_(0)
{
  if(v.term_!=0)
    term_=v.term_->clone();
}

const detail::Value& detail::Value::operator=(const detail::Value& v)
{
  if(term_)
    delete term_;
  if(v.term_!=0)
    term_=v.term_->clone();
  return *this;
}

detail::Evaluatable* detail::Value::clone() const
{
  return new Value(*this);
}

double detail::Value::value(const Parameters& p) const
{
  if (term_==0)
    boost::throw_exception(std::logic_error("No term present in Value"));
  return term_->value(p);
}

bool detail::Value::can_evaluate(const Parameters& p) const
{
  if (term_==0)
    boost::throw_exception(std::logic_error("No term present in Value"));
  return term_->can_evaluate(p);
}

void detail::Value::output(std::ostream& out) const
{
  if (term_==0)
    boost::throw_exception(std::logic_error("No term present in Value"));
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

detail::Number::Number(double val)
 : val_(val)
{}

double detail::Number::value(const Parameters&) const
{
  return val_;
}

bool detail::Number::can_evaluate(const Parameters&) const
{
  return true;
}

void detail::Number::output(std::ostream& out) const
{
  out << val_;
}

detail::Evaluatable* detail::Number::clone() const
{
  return new Number(*this);
}

detail::Symbol::Symbol(const std::string& name)
 : name_(name)
{}

double detail::Symbol::value(const Parameters& p) const
{
  if (name_=="Pi")
    return std::acos(-1.);
  if (p[name_].get<std::string>()=="Infinite recursion check" )
    boost::throw_exception(std::runtime_error("Infinite recursion when evaluating " + name_));
  Parameters parms(p);
  parms[name_]="Infinite recursion check";
  return evaluate(p[name_], p);
}

bool detail::Symbol::can_evaluate(const Parameters& p) const
{
  Parameters parms(p);
  parms[name_]=""; // set illegal to avoid infinite recursion
  return (name_=="Pi" || alps::can_evaluate(p[name_], parms));
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
 : Expression(in), name_(name)
{
  check_character(in,')',") expected after function call");
}

double detail::Function::value(const Parameters& p) const
{
  double val = Expression::value(p);
  if (name_=="sqrt")
    val = std::sqrt(val);
  else if (name_=="sin")
    val = std::sin(val);
  else if (name_=="cos")
    val = std::cos(val);
  else if (name_=="tan")
    val = std::tan(val);
  else if (name_=="exp")
    val = std::exp(val);
  else if (name_=="log")
    val = std::log(val);
  else
    boost::throw_exception(std::runtime_error("undefined function: " + name_));
  return val;
}

bool detail::Function::can_evaluate(const Parameters& p) const
{
  return Expression::can_evaluate(p) &&
         (name_=="sin" || name_=="cos" || name_=="tan" ||
          name_=="log" || name_=="exp" || name_=="sqrt");
}

void detail::Function::output(std::ostream& out) const
{
  out << name_ << "(";
  Expression::output(out);
  out << ")";
}

detail::Evaluatable* detail::Function::clone() const
{
  return new Function(*this);
}


Parameters evaluate(const Parameters& in) 
{
  Parameters out;
  for (Parameters::const_iterator p = in.begin(); p != in.end(); ++p) {
    const std::string name = p->key();
    const std::string value = static_cast<std::string>(p->value());
    Expression e(value);
    if (e.can_evaluate(in)) {
      // if value can be evaluated, then replace it by the evaluated value
      out[name] = e.value(in);
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
    if (i==0 || ops_[i-1]==MULTIPLY)
    {
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
  if(!term_)
    return boost::shared_ptr<Value>();
  boost::shared_ptr<detail::Evaluatable> term=term_->flatten_one();
  if (term) {
    boost::shared_ptr<Value> val(new Value(*this));
    val->term_=term->clone();
    return val;
  }
  else
    return boost::shared_ptr<Value>();
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
  flatten();
  return boost::shared_ptr<Expression>();
}

boost::shared_ptr<detail::Evaluatable> detail::Block::flatten_one()
{
  boost::shared_ptr<Expression> ex = flatten_one_expression();
  if (ex) {
    Block* block = new Block(*this);
    static_cast<Expression&>(*block) = *ex;
    return boost::shared_ptr<Evaluatable>(block);
  }
  else
    return boost::shared_ptr<Evaluatable>();
}

} // namespace alps
