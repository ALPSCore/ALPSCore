/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_SITETERM_H
#define ALPS_MODEL_SITETERM_H

#include <alps/model/siteoperator.h>

namespace alps {

class SiteTermDescriptor : public SiteOperator
{
public:
  typedef SiteOperator super_type;
  
  SiteTermDescriptor() : type_(-2) {}
  SiteTermDescriptor(const std::string& t, const std::string& s="") 
   : super_type(t,s), type_(-2) {}
  // template <class T>
  // SiteTermDescriptor(const T& t, const std::string& s="") 
  //   : super_type(t,s), type_(-2) {}
  SiteTermDescriptor(const XMLTag&, std::istream&);
  SiteTermDescriptor(SiteTermDescriptor const& t, std::string const& term, Parameters const& p, unsigned int type) 
   : super_type(t,term,p), type_(type) {}

  void write_xml(oxstream&) const;
  const SiteOperator& site_operator() const { return static_cast<const SiteOperator&>(*this);}
  bool match_type(int type) const { return type==type_;}

private:
  int type_;
};

template <class I, class T, class STATE>
class SiteOperatorEvaluator : public OperatorEvaluator<T>
{
private:
  typedef OperatorEvaluator<T> super_type;
  typedef SiteOperatorEvaluator<I,T,STATE> SELF_;

public:
  typedef STATE state_type;

  SiteOperatorEvaluator(const state_type& s, const SiteBasisDescriptor<I>& b,
                        const Parameters& p, const std::string sit="")
    : super_type(p), state_(s), basis_(b), fermionic_(false), site_(sit) {}
  bool can_evaluate(const std::string&,bool=false) const;
  bool can_evaluate_function(const std::string&, const expression::Expression<T>&, bool=false) const;
  bool can_evaluate_function(const std::string&, const std::vector<expression::Expression<T> >&, bool=false) const;
  expression::Expression<T> partial_evaluate(const std::string&,bool=false) const;
  expression::Expression<T> partial_evaluate_function(const std::string&, const expression::Expression<T>&, bool=false) const;
  expression::Expression<T> partial_evaluate_function(const std::string&, const std::vector<expression::Expression<T> >&, bool=false) const;
  const state_type& state() const { return state_;}
  bool fermionic() const { return fermionic_;}
  const std::string& site() const { return site_;}
  bool has_operator(const std::string& n) const { return basis_.has_operator(n);}
private:

  mutable state_type state_;
  const SiteBasisDescriptor<I>& basis_;
  mutable bool fermionic_;
  std::string site_;
};


template <class I, class T, class STATE>
bool SiteOperatorEvaluator<I,T,STATE>::can_evaluate(const std::string& name,bool isarg) const
{
  if (basis_.has_operator(name)) {
    SELF_ eval(*this);
    return eval.partial_evaluate(name,isarg).can_evaluate(expression::ParameterEvaluator<T>(*this),isarg);
  }
  return expression::ParameterEvaluator<T>::can_evaluate(name,isarg);
}

template <class I, class T, class STATE>
bool SiteOperatorEvaluator<I,T,STATE>::can_evaluate_function(const std::string& name,const expression::Expression<T>& arg, bool isarg) const
{
  if (arg==site() && basis_.has_operator(name))
    return can_evaluate(name,isarg);
  else
    return expression::ParameterEvaluator<T>::can_evaluate_function(name,arg,isarg);
}

template <class I, class T, class STATE>
bool SiteOperatorEvaluator<I,T,STATE>::can_evaluate_function(const std::string& name,const std::vector<expression::Expression<T> >& args, bool isarg) const
{
  if (args.size()==1)
    return can_evaluate_function(name,args[0],isarg);
  else
    return expression::ParameterEvaluator<T>::can_evaluate_function(name,args,isarg);
}

template <class I, class T, class STATE>
expression::Expression<T> SiteOperatorEvaluator<I,T,STATE>::partial_evaluate(const std::string& name,bool isarg) const
{
  if (basis_.has_operator(name)) {
    expression::Expression<T> e;
    bool fermionic;
    boost::tie(state_,e,fermionic) = basis_.apply(name,state_, expression::ParameterEvaluator<T>(*this),isarg);
    if (fermionic)
      fermionic_=!fermionic_;
    return e;
  }
  else
    return super_type::partial_evaluate(name,isarg);
}

template <class I, class T, class STATE>
expression::Expression<T> SiteOperatorEvaluator<I,T,STATE>::partial_evaluate_function(const std::string& name,const expression::Expression<T>& arg,bool isarg) const
{
  if (arg==site() && basis_.has_operator(name)) {  // evaluate operator
    expression::Expression<T> e;
    bool fermionic;
    boost::tie(state_,e,fermionic) = basis_.apply(name,state_, expression::ParameterEvaluator<T>(*this),isarg);
    if (fermionic)
      fermionic_=!fermionic_;
    return e;
  }
  else
    return expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,arg,isarg);
}

template <class I, class T, class STATE>
expression::Expression<T> SiteOperatorEvaluator<I,T,STATE>::partial_evaluate_function(const std::string& name,const std::vector<expression::Expression<T> >& args,bool isarg) const
{
  if (args.size()==1)
    return partial_evaluate_function(name,args[0],isarg);
  else
    return expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,args,isarg);
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::SiteTermDescriptor& q)
{
  q.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::SiteTermDescriptor& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
