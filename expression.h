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
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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

class Factor;
class Term;

namespace detail {

class Evaluatable {
public:
  Evaluatable() {}
  virtual ~Evaluatable() {}
  double value() const;
  virtual double value(const Evaluator&) const=0;
  operator double() const;
  bool can_evaluate() const;
  virtual bool can_evaluate(const Evaluator&) const=0;
  virtual void output(std::ostream&) const =0;
  virtual Evaluatable* clone() const=0;
  virtual boost::shared_ptr<Evaluatable> flatten_one();
  virtual Evaluatable* partial_evaluate_replace(const Evaluator& p);
  virtual bool is_single_term() const;
  virtual Term term() const;
  virtual bool depends_on(const std::string&) const;
};

} // end namespace detail

class Factor : public detail::Evaluatable {
public:
  Factor(std::istream&, bool inverse=false);
  Factor(double x);
  Factor(const std::string& s);
  Factor(const Factor&);
  Factor(const detail::Evaluatable& e)  : term_(e.clone()), is_inverse_(false) {}
  const Factor& operator=(const Factor&);
  double value(const Evaluator& p) const;
  void output(std::ostream&) const;
  bool can_evaluate(const Evaluator& p) const;
  detail::Evaluatable* clone() const;
  boost::shared_ptr<Factor> flatten_one_value();
  bool is_inverse() const { return is_inverse_;}
  void partial_evaluate(const Evaluator& p);
  bool is_single_term() const;
  Term term() const;
  bool depends_on(const std::string&) const;
private:
  boost::shared_ptr<detail::Evaluatable> term_;
  bool is_inverse_;
};


class Term : public detail::Evaluatable {
public:
  Term(std::istream&, bool =false);
  Term() : is_negative_(false) {}
  Term(double x) : is_negative_(false), terms_(1,Factor(x)) {}
  Term(const detail::Evaluatable& e) : is_negative_(false), terms_(1,Factor(e)) {}
  virtual ~Term() {}
  double value(const Evaluator& p) const;
  bool can_evaluate(const Evaluator& p) const;
  void output(std::ostream&) const;
  detail::Evaluatable* clone() const;
  bool is_negative() const { return is_negative_;}
  boost::shared_ptr<Term> flatten_one_term();
  void partial_evaluate(const Evaluator& p);
  inline operator std::string () const;
  const Term& operator*=(const Factor& v) { terms_.push_back(v); return *this;}
  const Term& operator*=(const std::string& s) { return operator*=(Factor(s));}
  void simplify();
  typedef std::vector<Factor>::const_iterator factor_iterator;
  virtual std::pair<factor_iterator,factor_iterator> factors() const;
  bool depends_on(const std::string&) const;
private:
  void negate() { is_negative_ = !is_negative_;}
  bool is_negative_;
  std::vector<Factor> terms_;
};


class Expression : public detail::Evaluatable {
public:
  typedef std::vector<Term>::const_iterator term_iterator;
  Expression() {}
  Expression(const std::string&);
  Expression(std::istream&);
  Expression(double val) : terms_(1,Term(val)) {}
  Expression(const detail::Evaluatable& e) : terms_(1,Term(e)) {}
  operator bool() const;
  double value(const Evaluator& p=Evaluator()) const;
  bool can_evaluate(const Evaluator& p=Evaluator()) const;
  void partial_evaluate(const Evaluator& p=Evaluator());
  void partial_evaluate(const Parameters& p) { partial_evaluate(ParameterEvaluator(p));}
  void output(std::ostream&) const;
  detail::Evaluatable* clone() const;
  std::pair<term_iterator,term_iterator> terms() const 
  { return std::make_pair(terms_.begin(),terms_.end());}
  void flatten(); // multiply out all blocks
  boost::shared_ptr<Expression> flatten_one_expression();
  const Expression& operator +=(const Term& term) { terms_.push_back(term); return *this;}
  const Expression& operator +=(const Expression& e);
  inline operator std::string () const;
  void simplify();
  bool is_single_term() const;
  Term term() const;
  bool depends_on(const std::string&) const;
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

Expression operator+(const Expression& ex1,const Expression& ex2);
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

inline bool operator<(const alps::Term& e1, const alps::Term& e2)
{
  return static_cast<std::string>(e1) < static_cast<std::string>(e2);
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
