/***************************************************************************
* ALPS++ library
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
#include <alps/parameters.h>
#include <alps/parser/parser.h>
#include <boost/smart_ptr.hpp>
#include <string>
#include <vector>

namespace alps {

class Expression;

namespace detail {

class Evaluatable {
public:
  Evaluatable() {}
  virtual ~Evaluatable() {}
  double value() const;
  virtual double value(const Parameters&) const=0;
  bool can_evaluate() const;
  virtual bool can_evaluate(const Parameters&) const=0;
  virtual void output(std::ostream&) const =0;
  virtual Evaluatable* clone() const=0;
  virtual boost::shared_ptr<Evaluatable> flatten_one();
};

class Value : public Evaluatable {
public:
  Value(std::istream&);
  virtual ~Value();
  Value(const Value&);
  const Value& operator=(const Value&);
  double value(const Parameters& p) const;
  void output(std::ostream&) const;
  bool can_evaluate(const Parameters& p) const;
  Evaluatable* clone() const;
  boost::shared_ptr<Value> flatten_one_value();
private:
  Evaluatable* term_;
};

} // end namespace detail

class Term : public detail::Evaluatable {
public:
  Term(std::istream&, bool =false);
  virtual ~Term() {}
  double value(const Parameters& p) const;
  bool can_evaluate(const Parameters& p) const;
  void output(std::ostream&) const;
  detail::Evaluatable* clone() const;
  bool is_negative() const { return is_negative_;}
  boost::shared_ptr<Term> flatten_one_term();
private:
  bool is_negative_;
  std::vector<detail::Value> terms_;
  enum op {MULTIPLY,DIVIDE};
  std::vector<op> ops_;
};


class Expression : public detail::Evaluatable {
public:
  typedef std::vector<Term>::const_iterator term_iterator;
  Expression() {}
  Expression(const std::string&);
  Expression(std::istream&);
  double value(const Parameters& p=Parameters()) const;
  bool can_evaluate(const Parameters& p=Parameters()) const;
  void output(std::ostream&) const;
  detail::Evaluatable* clone() const;
  std::pair<term_iterator,term_iterator> terms() const 
  { return std::make_pair(terms_.begin(),terms_.end());}
  void flatten(); // multiply out all blocks
  boost::shared_ptr<Expression> flatten_one_expression();
private:
  std::vector<Term> terms_;
};

// evaluate all the parameters as far as possible
extern Parameters evaluate(const Parameters& in);

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline bool can_evaluate(const alps::StringValue& v, const alps::Parameters& p = alps::Parameters())
{
  return v.valid() && Expression(v).can_evaluate(p);
}

inline double evaluate(const alps::StringValue& v, const alps::Parameters& p = alps::Parameters())
{
  return Expression(v).value(p);
}

// inline bool can_evaluate(const StringValue& v)
// {
//   return v.valid() && Expression(v).can_evaluate();
// }

// inline double evaluate(const StringValue& v)
// {
//   return Expression(v).value();
// }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

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

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_EXPRESSION_H
