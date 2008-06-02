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

#ifndef ALPS_EXPRESSION_TERM_H
#define ALPS_EXPRESSION_TERM_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/factor.h>
#include <boost/call_traits.hpp>

namespace alps {
namespace expression {

template<class T>
class Term : public Evaluatable<T> {
public:
  typedef T value_type;
  typedef typename std::vector<Factor<T> >::const_iterator factor_iterator;

  Term(std::istream& is, bool negate = false);
  Term() : is_negative_(false) {}
  Term(typename boost::call_traits<value_type>::param_type x)
    : is_negative_(false), terms_(1,Factor<T>(x)) {}
  Term(const Evaluatable<T>& e)
    : is_negative_(false), terms_(1,Factor<T>(e)) {}
  Term(const std::pair<T,Term<T> >&);
  virtual ~Term() {}

  value_type value(const Evaluator<T>& =Evaluator<T>(), bool=false) const;

  bool can_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  void output(std::ostream&) const;
  Evaluatable<T>* clone() const { return new Term<T>(*this); }
  bool is_negative() const { return is_negative_;}
  boost::shared_ptr<Term> flatten_one_term();
  void partial_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false);
  std::pair<T,Term<T> > split() const;
  
  const Term& operator*=(const Factor<T>& v)
  {
    terms_.push_back(v);
    return *this;
  }
  const Term& operator*=(const std::string& s)
  {
    return operator*=(Factor<T>(s));
  }
  
  void simplify();
  void remove_spurious_parentheses();

  virtual std::pair<factor_iterator,factor_iterator> factors() const
  {
    return std::make_pair(terms_.begin(),terms_.end());
  }

  bool depends_on(const std::string&) const;

  int num_factors() const {return terms_.size(); }
  void negate() { is_negative_ = !is_negative_;}

private:
  bool is_negative_;
  std::vector<Factor<T> > terms_;
};

template <class T>
struct term_less {
  bool operator()(const Term<T>& x, const Term<T>& y) {
    return x.split().second < y.split().second;
  }
};

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
  partial_evaluate(Evaluator<T>(false));
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
    for (unsigned int i = 0; i < terms_.size() && is_nonzero(val); ++i)
      val *= terms_[i].value(p,isarg);
}
  else {
    for (int i = int(terms_.size())-1; i >= 0 && is_nonzero(val); --i) {
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
      for (unsigned int i=0; i<terms_.size(); ++i) {
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
      for (int i = int(terms_.size())-1; i >= 0; --i) {
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
  for (unsigned int i=0;i<terms_.size();++i)
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
  for (unsigned int i=1;i<terms_.size();++i) {
    os << " " << (terms_[i].is_inverse() ? "/" : "*") << " ";
    terms_[i].output(os);
  }
}

template<class T>
Term<T>::Term(const std::pair<T,Term<T> >& t)
 :  is_negative_(false),terms_(t.second.terms_)
{
  terms_.insert(terms_.begin(), Factor<T>(t.first));
  partial_evaluate(Evaluator<T>(false));
}

template<class T>
std::pair<T,Term<T> > Term<T>::split() const
{
  Term<T> t(*this);
  t.partial_evaluate(Evaluator<T>(false));
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


template<class T>
boost::shared_ptr<Term<T> > Term<T>::flatten_one_term()
{
  for (unsigned int i=0;i<terms_.size();++i)
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

}
}


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace expression {
#endif

template<class T>
inline bool operator==(const alps::expression::Term<T>& ex1, const alps::expression::Term<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) ==
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator==(const alps::expression::Term<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) == s;
}

template<class T>
inline bool operator==(const std::string& s, const alps::expression::Term<T>& ex)
{
  return ex == s;
}

template<class T>
inline bool operator<(const alps::expression::Term<T>& ex1, const alps::expression::Term<T>& ex2)
{
  return (boost::lexical_cast<std::string>(ex1) <
          boost::lexical_cast<std::string>(ex2));
}

template<class T>
inline bool operator<(const alps::expression::Term<T>& ex, const std::string& s)
{
  return boost::lexical_cast<std::string>(ex) < s;
}

template<class T>
inline bool operator<(const std::string& s, const alps::expression::Term<T>& ex)
{
  return s < boost::lexical_cast<std::string>(ex);
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace expression
} // end namespace alps
#endif

#endif
