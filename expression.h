/***************************************************************************
* ALPS library
*
* alps/expression.h   A class to evaluate expressions
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

#ifndef ALPS_EXPRESSION_H
#define ALPS_EXPRESSION_H

#include <alps/config.h>
#include <alps/evaluator.h>
#include <alps/parser/parser.h>
#include <boost/smart_ptr.hpp>
#include <string>
#include <vector>

namespace alps {

namespace detail {

class Evaluatable {
public:
  Evaluatable() {}
  virtual ~Evaluatable() {}
  double value() const;
  virtual double value(const Evaluator&) const=0;
  bool can_evaluate() const;
  virtual bool can_evaluate(const Evaluator&) const=0;
  virtual void output(std::ostream&) const =0;
  virtual Evaluatable* clone() const=0;
  virtual boost::shared_ptr<Evaluatable> flatten_one();
  virtual Evaluatable* partial_evaluate_replace(const Evaluator& p);
};

class Value : public Evaluatable {
public:
  Value(std::istream&, bool inverse=false);
  Value(double x);
  Value(const Value&);
  Value(const Evaluatable& e)  : term_(e.clone()), is_inverse_(false) {}
  const Value& operator=(const Value&);
  double value(const Evaluator& p) const;
  void output(std::ostream&) const;
  bool can_evaluate(const Evaluator& p) const;
  Evaluatable* clone() const;
  boost::shared_ptr<Value> flatten_one_value();
  bool is_inverse() const { return is_inverse_;}
  void partial_evaluate(const Evaluator& p);
private:
  boost::shared_ptr<Evaluatable> term_;
  bool is_inverse_;
};

} // end namespace detail

class Term : public detail::Evaluatable {
public:
  Term(std::istream&, bool =false);
  Term(double x) : is_negative_(false), terms_(1,detail::Value(x)) {}
  Term(const Evaluatable& e) : is_negative_(false), terms_(1,detail::Value(e)) {}
  virtual ~Term() {}
  double value(const Evaluator& p) const;
  bool can_evaluate(const Evaluator& p) const;
  void output(std::ostream&) const;
  detail::Evaluatable* clone() const;
  bool is_negative() const { return is_negative_;}
  boost::shared_ptr<Term> flatten_one_term();
  void partial_evaluate(const Evaluator& p);
  inline operator std::string () const;
private:
  bool is_negative_;
  std::vector<detail::Value> terms_;
};


class Expression : public detail::Evaluatable {
public:
  typedef std::vector<Term>::const_iterator term_iterator;
  Expression() {}
  Expression(const std::string&);
  Expression(std::istream&);
  Expression(double val) : terms_(1,Term(val)) {}
  Expression(const Evaluatable& e) : terms_(1,Term(e)) {}
  double value(const Evaluator& p=Evaluator()) const;
  bool can_evaluate(const Evaluator& p=Evaluator()) const;
  void partial_evaluate(const Evaluator& p=Evaluator());
  void output(std::ostream&) const;
  detail::Evaluatable* clone() const;
  std::pair<term_iterator,term_iterator> terms() const 
  { return std::make_pair(terms_.begin(),terms_.end());}
  void flatten(); // multiply out all blocks
  boost::shared_ptr<Expression> flatten_one_expression();
  const Expression& operator +=(const Term& term) { terms_.push_back(term); return *this;}
  const Expression& operator +=(const Expression& e);
  inline operator std::string () const;
private:
  std::vector<Term> terms_;
};

// evaluate all the parameters as far as possible
extern Parameters evaluate(const Parameters& in);

inline bool can_evaluate(const StringValue& v, const Evaluator& eval= Evaluator())
{
  return v.valid() && Expression(static_cast<std::string>(v)).can_evaluate(eval);
}

inline double evaluate(const StringValue& v, const Evaluator& p = Evaluator())
{
  return Expression(static_cast<std::string>(v)).value(p);
}

inline bool can_evaluate(const StringValue& v, const Parameters& p)
{
  return can_evaluate(v,ParameterEvaluator(p));
}

inline double evaluate(const StringValue& v, const Parameters& p)
{
  return evaluate(v,ParameterEvaluator(p));
}

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace detail {
#endif

inline std::ostream& operator<<(std::ostream& os,
				const alps::detail::Evaluatable& e)
{
  e.output(os);
  return os;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace detail
#endif

inline std::istream& operator>>(std::istream& is, alps::Expression& e)
{
  std::string s;
  is >> s;
  e = Expression(s);
  return is;
}


inline bool operator==(const std::string& s, const alps::Expression& e)
{
  return s==static_cast<std::string>(e);
}

inline bool operator==(const alps::Expression& e, const std::string& s)
{
  return static_cast<std::string>(e)==s;
}

inline bool operator==(const alps::Expression& e1, const alps::Expression& e2)
{
  return static_cast<std::string>(e1) == static_cast<std::string>(e2);
}

inline bool operator==(const alps::Term& e, const std::string& s)
{
  return s==static_cast<std::string>(e);
}

inline bool operator==( const std::string& s, const alps::Term& e)
{
  return s==static_cast<std::string>(e);
}

inline bool operator==(const alps::Term& e1, const alps::Term& e2)
{
  return static_cast<std::string>(e1) == static_cast<std::string>(e2);
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

alps::Expression::operator std::string() const 
{ 
  return boost::lexical_cast<std::string,Expression>(*this);
}

alps::Term::operator std::string() const 
{ 
  return boost::lexical_cast<std::string,Term>(*this);
}


#endif // ALPS_EXPRESSION_H
