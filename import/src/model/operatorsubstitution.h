/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2005 by Matthias Troyer <troyer@comp-phys.org>
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

#ifndef ALPS_MODEL_OPERATORSUBSTITUTION_H
#define ALPS_MODEL_OPERATORSUBSTITUTION_H

#include <alps/model/siteoperator.h>
#include <alps/model/bondoperator.h>
#include <alps/model/modellibrary.h>

namespace alps {

class ModelLibrary;

template <class T=std::complex<double> >
class OperatorSubstitution : public expression::ParameterEvaluator<T>
{
private:
  typedef expression::ParameterEvaluator<T> super_type;

public:
  typedef std::map<std::string,SiteOperator> SiteOperatorMap;
  typedef std::map<std::string,BondOperator> BondOperatorMap;
  
  OperatorSubstitution(const ModelLibrary& lib, const Parameters& p, const std::vector<std::string>& s) 
   : super_type(p,false), sitemap_(lib.site_operators()), bondmap_(lib.bond_operators()), sites_(s) {}
  expression::Expression<T> partial_evaluate(const std::string& name, bool=false) const;
  bool can_evaluate_function(const std::string&, const std::vector<expression::Expression<T> >&, bool=false) const;
  expression::Expression<T> partial_evaluate_function(const std::string&, const std::vector<expression::Expression<T> >&, bool=false) const;

  void substitute_arguments(const std::map<std::string,std::string>& p) { subs_=p;}
  void set_sites(const std::vector<std::string>& s) { sites_ = s;}

private:
  bool correct_arguments(const std::vector<expression::Expression<T> >&) const;

  const SiteOperatorMap& sitemap_;
  const BondOperatorMap& bondmap_;
  std::vector<std::string> sites_;
  std::map<std::string,std::string> subs_;
};

template <class T>
expression::Expression<T> OperatorSubstitution<T>::partial_evaluate(const std::string& name, bool isarg) const
{
  std::map<std::string,std::string>::const_iterator it=subs_.find(name);
  return it!=subs_.end() ? expression::Expression<T>(it->second) : super_type::partial_evaluate(name,isarg);
}

template <class T>
bool OperatorSubstitution<T>::correct_arguments(const std::vector<expression::Expression<T> >& args) const
{
  std::vector<expression::Expression<T> > evalargs(args);
  expression::ParameterEvaluator<T>::partial_evaluate_expressions(evalargs,true);
  for (typename std::vector<expression::Expression<T> >::const_iterator it=evalargs.begin();it!=evalargs.end();++it)
    if (std::find(sites_.begin(),sites_.end(),*it) == sites_.end())
      return false;
  return true;
}


template <class T>
bool OperatorSubstitution<T>::can_evaluate_function(const std::string& name,const std::vector<expression::Expression<T> >& args, bool isarg) const
{
  return expression::ParameterEvaluator<T>::can_evaluate_function(name,args,isarg);
}


template <class T>
expression::Expression<T> OperatorSubstitution<T>::partial_evaluate_function(const std::string& name,const std::vector<expression::Expression<T> >& args, bool isarg) const
{
  std::vector<expression::Expression<T> > evalargs(args);
  expression::ParameterEvaluator<T>::partial_evaluate_expressions(evalargs,true);
  bool correctargs=correct_arguments(evalargs);
    
  if (correctargs && args.size()==2) {
    BondOperatorMap::const_iterator fun = bondmap_.find(name);
    if (fun != bondmap_.end()) {
      std::map<std::string,std::string> p;
      p[fun->second.source()]=boost::lexical_cast<std::string>(evalargs[0]);
      p[fun->second.target()]=boost::lexical_cast<std::string>(evalargs[1]);
      OperatorSubstitution subs(*this);
      subs.substitute_arguments(p);
      expression::Expression<T> e(fun->second.term());
      e.partial_evaluate(subs,isarg);
      return e;
    }
    else
      return expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,args,isarg);
  }
  else if (correctargs && args.size()==1) {
    SiteOperatorMap::const_iterator fun = sitemap_.find(name);
    if (fun != sitemap_.end()) {
      std::map<std::string,std::string> p;
      p[fun->second.site()]=boost::lexical_cast<std::string>(evalargs[0]);
      OperatorSubstitution subs(*this);
      subs.substitute_arguments(p);
      expression::Expression<T> e(fun->second.term());
      e.partial_evaluate(subs,isarg);
      return e;
    }
    else
      return expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,args,isarg);
  }
  else 
    return expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,args,isarg);
}

} // namespace alps

#endif
