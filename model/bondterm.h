/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2009 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_BONDTERM_H
#define ALPS_MODEL_BONDTERM_H

#include <alps/model/bondoperator.h>
#include <alps/model/siteoperator.h>
#include <alps/numeric/is_nonzero.hpp>

namespace alps {


class BondTermDescriptor : public BondOperator
{
public:
  BondTermDescriptor() : BondOperator(), type_(-2) {}
  BondTermDescriptor(const std::string& s, const std::string& t) : BondOperator(s,t), type_(-2) {}
  // template <class T>
  // BondTermDescriptor(const T& term) : BondOperator(term, "i", "j"), type_(-2) {}
  // template <class T>
  // BondTermDescriptor(const T& term, const std::string& s) : BondOperator(term, s, "j"), type_(-2) {}
  // template <class T>
  // BondTermDescriptor(const T& term, const std::string& s, const std::string& t)
  //   : BondOperator(term, s, t), type_(-2) {}
  BondTermDescriptor(const std::string& term, const std::string& s, const std::string& t)
    : BondOperator(term, s, t), type_(-2) {}

  BondTermDescriptor(const XMLTag&, std::istream&);

  BondTermDescriptor(BondTermDescriptor const& t, std::string const& term, Parameters const& p, unsigned int type)
   : BondOperator(t,term,p), type_(type) {}

  const BondOperator& bond_operator() const { return static_cast<const BondOperator&>(*this);}
  void write_xml(oxstream&) const;

  bool match_type(int type) const { return type_==-1 || type==type_;}
private:
  int type_;
};

template <class I, class T, class STATE1, class STATE2>
class BondOperatorEvaluator : public OperatorEvaluator<T>
{
private:
  typedef OperatorEvaluator<T> super_type;
  typedef BondOperatorEvaluator<I, T, STATE1, STATE2> SELF_;

public:
  BondOperatorEvaluator(const STATE1& s1, const STATE2& s2,
                        const SiteBasisDescriptor<I>& b1,
                        const SiteBasisDescriptor<I>& b2,
                        const std::string& site1, const std::string& site2,
                        const Parameters& p)
    : super_type(p), site1_(s1,b1,p,site1), site2_(s2,b2,p,site2) {}

  BondOperatorEvaluator(const SiteOperatorEvaluator<I,T,STATE1>& s1, const SiteOperatorEvaluator<I,T,STATE2>& s2,const Parameters& p)
    : super_type(p), site1_(s1), site2_(s2) {}

  bool can_evaluate_function(const std::string&, const expression::Expression<T>&, bool=false) const;
  expression::Expression<T> partial_evaluate_function(const std::string& name,
                                        const expression::Expression<T>& argument,bool=false) const;
  std::pair<STATE1,STATE2> state() const { return std::make_pair(site1_.state(),site2_.state());}
  bool fermionic() const {
    if (site1_.fermionic()!=site2_.fermionic())
      boost::throw_exception(std::runtime_error("Bond term contains unphysical single fermion creation operator"));
    return site1_.fermionic();
  }

  bool has_operator(const std::string& name, const expression::Expression<T>& arg) const
  {
    return (arg==site1_.site() && site1_.has_operator(name)) ||
           (arg==site2_.site() && site2_.has_operator(name));
  }

private:
  SiteOperatorEvaluator<I,T,STATE1> site1_;
  SiteOperatorEvaluator<I,T,STATE2> site2_;
};


template <class I, class T, class STATE1, class STATE2>
bool BondOperatorEvaluator<I,T,STATE1, STATE2>::can_evaluate_function(const std::string& name, const expression::Expression<T>& arg, bool isarg) const
{
  if (has_operator(name,arg)) {
    SELF_ eval(*this);
    return eval.partial_evaluate_function(name,arg,isarg).can_evaluate(expression::ParameterEvaluator<T>(*this),isarg);
  }
  else
    return expression::ParameterEvaluator<T>::can_evaluate_function(name,arg,isarg);
}

template <class I, class T, class STATE1, class STATE2>
expression::Expression<T> BondOperatorEvaluator<I,T,STATE1,STATE2>::partial_evaluate_function(const std::string& name, const expression::Expression<T>& arg, bool isarg) const
{
  expression::Expression<T> e;
  if (has_operator(name,arg)) {  // evaluate operator
    bool f;
    if (arg==site1_.site()) {
      f = site1_.fermionic();
      e =  site1_.partial_evaluate_function(name,site1_.site(),isarg);
      if (f != site1_.fermionic() && numeric::is_nonzero(e) && site2_.fermionic()) // for normal ordering
        e.negate();
    }
    else  if (arg==site2_.site())
      e = site2_.partial_evaluate_function(name,site2_.site(),isarg);
    else
      e = expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,arg,isarg);
  }
  else
    e = expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,arg,isarg);
  return e;
}



} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::BondTermDescriptor& q)
{
  q.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::BondTermDescriptor& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
