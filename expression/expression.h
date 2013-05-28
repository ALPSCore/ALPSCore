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

#ifndef ALPS_EXPRESSION_EXPRESSION_H
#define ALPS_EXPRESSION_EXPRESSION_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/term.h>
#include <alps/expression/term.h>
#include <boost/call_traits.hpp>
#include <alps/numeric/is_zero.hpp>

namespace alps {
namespace expression {

template<class T>
class Expression : public Evaluatable<T> {
public:
  typedef T value_type;
  typedef Term<T> term_type;
  typedef typename std::vector<Term<T> >::const_iterator term_iterator;

  Expression() {}
  Expression(const std::string& str) { parse(str); }
  Expression(std::istream& in) { parse(in); }
  Expression(typename boost::call_traits<value_type>::param_type val)
    : terms_(1,Term<T>(val)) {}
#ifndef BOOST_NO_SFINAE
  template<class U>
  Expression(U val, typename boost::enable_if<boost::is_arithmetic<U> >::type* = 0) : terms_(1,Term<T>(value_type(val))) {}
#else
  Expression(int val) : terms_(1,Term<T>(value_type(val))) {}
#endif
  Expression(const Evaluatable<T>& e) : terms_(1,Term<T>(e)) {}
  Expression(const Term<T>& e) : terms_(1,e) {}
  virtual ~Expression() {}

  value_type value(const Evaluator<T>& = Evaluator<T>(), bool=false) const;
  value_type value(const Parameters& p) const {
    return value(ParameterEvaluator<T>(p));
  }

  bool can_evaluate(const Evaluator<T>& = Evaluator<T>(), bool=false) const;
  bool can_evaluate(const Parameters& p) const
  {
    return can_evaluate(ParameterEvaluator<T>(p));
  }
  void partial_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false);
  void partial_evaluate(const Parameters& p) {
    partial_evaluate(ParameterEvaluator<T>(p));
  }

  void sort();
  void output(std::ostream& os) const;

  Evaluatable<T>* clone() const { return new Expression<T>(*this); }

  std::pair<term_iterator,term_iterator> terms() const
  {
    return std::make_pair(terms_.begin(),terms_.end());
  }

  void flatten(); // multiply out all blocks

  boost::shared_ptr<Expression> flatten_one_expression();

  const Expression& operator+=(const Term<T>& term)
  {
    terms_.push_back(term);
    partial_evaluate(Evaluator<T>(false));
    return *this;
  }

  const Expression& operator-=(Term<T> term)
  {
    term.negate();
    terms_.push_back(term);
    partial_evaluate(Evaluator<T>(false));
    return *this;
  }

  const Expression& operator+=(const Expression& e)
  {
    std::copy(e.terms_.begin(),e.terms_.end(),std::back_inserter(terms_));
    partial_evaluate(Evaluator<T>(false));
    return *this;
  }

  const Expression& operator-=(Expression const& e)
  {
    return operator+=(-e);
  }

  const Expression& operator*=(const Expression<T>& e)
  {
    Term<T> newt(Factor<T>(Block<T>(*this)));
    newt *= Factor<T>(Block<T>(e));
    terms_.clear();
    newt.remove_superfluous_parentheses();
    terms_.push_back(newt);
    partial_evaluate(Evaluator<T>(false));
    return *this;
  }
  
  
  void simplify();
  void remove_superfluous_parentheses();

  bool has_no_term()    const { return terms_.empty(); }
  bool is_single_term() const { return terms_.size() == 1; }
  Term<T> term() const;
  Term<T> zeroth_term() const { return terms_[0]; }
  bool depends_on(const std::string&) const;

  Expression<T> expression_dependent_on(const std::string&) const;
  Expression<T> expression_dependent_only_on(const std::string&) const;

  void parse(const std::string& str);
  bool parse(std::istream& is);

  Expression operator-() const { Expression e(*this); e.negate(); return e;}
  const Expression& negate() 
  {
    for (typename std::vector<Term<T> >::iterator it=terms_.begin();it!=terms_.end();++it)
      it->negate();
    return *this;
  } 
private:
  std::vector<Term<T> > terms_;
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
void Expression<T>::remove_superfluous_parentheses()
{
  for (typename std::vector<Term<T> >::iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    it->remove_superfluous_parentheses();
}

template<class T>
void Expression<T>::simplify()
{
  partial_evaluate(Evaluator<T>(false));
  for (typename std::vector<Term<T> >::iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    it->simplify();
  sort();
  partial_evaluate(Evaluator<T>(false));
}

template<class T>
Term<T> Expression<T>::term() const
{
  if (!is_single_term())
    boost::throw_exception(std::logic_error("Called term() for multi-term expression"));
  return terms_[0];
}

template<class T>
Expression<T> Expression<T>::expression_dependent_on(const std::string& str) const
{
  Expression<T> e;
  for (typename std::vector<Term<T> >::const_iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    if (it->depends_on(str))
      e += (*it);
  return e;
} 

template<class T>
Expression<T> Expression<T>::expression_dependent_only_on(const std::string& str) const
{
  Expression<T> e;
  for (typename std::vector<Term<T> >::const_iterator it=terms_.begin();
       it!=terms_.end(); ++it)
    if (it->depends_only_on(str))
      e += (*it);
  return e;
}

template<class T>
void Expression<T>::parse(const std::string& str)
{
  std::istringstream in(str);
  if (!parse(in))
    boost::throw_exception(std::runtime_error("Did not parse to end of string '" + str + "'"));
}

template<class T>
bool Expression<T>::parse(std::istream& is)
{
  terms_.clear();
  bool negate=false;
  char c;
  is >> c;
  if (is.eof())
    return true;
  if (c=='-')
    negate=true;
  else if (c=='+')
    negate=false;
  else
    is.putback(c);
  terms_.push_back(Term<T>(is,negate));
  while(true) {
    if(!(is >> c))
      return true;
    if (is.eof())
      return true;
    if (c=='-')
      negate=true;
    else if (c=='+')
      negate=false;
    else {
      is.putback(c);
      return false;
    }
    terms_.push_back(Term<T>(is,negate));
  }
}

template <class T>
void Expression<T>::sort()
{
  partial_evaluate(Evaluator<T>(false));
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
      it = prev;
      ++it;
    }
    else {
      if (added && numeric::is_zero(prev_term.first))
        terms_.erase(prev);
      else {
        prev=it;
        ++it;
      }
      prev_term=current_term;
    }
    added=false;
  }
  if (added && numeric::is_zero(prev_term.first))
    terms_.erase(prev);
}

template<class T>
typename Expression<T>::value_type Expression<T>::value(const Evaluator<T>& p, bool isarg) const
{
  if (terms_.size()==0)
    return value_type(0.);
  value_type val=terms_[0].value(p);
  for (unsigned int i=1;i<terms_.size();++i)
    val += terms_[i].value(p,isarg);
  return val;
}

template<class T>
bool Expression<T>::can_evaluate(const Evaluator<T>& p, bool isarg) const
{
  if (terms_.size()==0)
    return true;
  bool can=true;

  for (unsigned int i=0;i<terms_.size();++i)
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
    for (unsigned int i=0; i<terms_.size(); ++i) {
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
    for (unsigned int i=1;i<terms_.size();++i) {
      if(!terms_[i].is_negative())
        os << " + ";
      terms_[i].output(os);
    }
  }
}

template<class T>
void Expression<T>::flatten()
{
  unsigned int i=0;
  while (i<terms_.size()) {
    boost::shared_ptr<Term<T> > term = terms_[i].flatten_one_term();
    if (term)
      terms_.insert(terms_.begin()+i,*term);
    else
      ++i;
  }
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

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace expression {
#endif

template<class T>
inline alps::expression::Expression<T> operator+(const alps::expression::Expression<T>& ex1, const alps::expression::Expression<T>& ex2)
{
  alps::expression::Expression<T> ex(ex1);
  ex += ex2;
  return ex;
}

template<class T>
inline std::istream& operator>>(std::istream& is, alps::expression::Expression<T>& e)
{
  std::string s;
  is >> s;
  e.parse(s);
  return is;
}

template<class T>
inline bool operator==(const alps::expression::Expression<T>& ex1, const alps::expression::Expression<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) ==
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator==(const alps::expression::Expression<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) == s;
}

template<class T>
inline bool operator==(const std::string& s, const alps::expression::Expression<T>& ex)
{
  return ex == s;
}

template<class T>
inline bool operator!=(const alps::expression::Expression<T>& ex1, const alps::expression::Expression<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) !=
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator!=(const alps::expression::Expression<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) != s;
}

template<class T>
inline bool operator!=(const std::string& s, const alps::expression::Expression<T>& ex)
{
  return ex != s;
}

template<class T>
inline bool operator<(const alps::expression::Expression<T>& ex1, const alps::expression::Expression<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) <
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator<(const alps::expression::Expression<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) < s;
}

template<class T>
inline bool operator<(const std::string& s, const alps::expression::Expression<T>& ex)
{
  return s < boost::lexical_cast<std::string>(ex);
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace expression
} // end namespace alps
#endif

#endif // ! ALPS_EXPRESSION_IMPL_H
