/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_EXPRESSION_PARAMETEREVALUATOR_H
#define ALPS_EXPRESSION_PARAMETEREVALUATOR_H

#include <alps/expression/evaluator.h>
#include <alps/expression/evaluate.h>

namespace alps {
namespace expression {

//
// implementation of ParameterEvaluator<T>
//

template<class T>
bool ParameterEvaluator<T>::can_evaluate(const std::string& name, bool isarg) const
{
  if (evaluate_helper<T>::can_evaluate_symbol(name,isarg)) return true;
  if (!parms_.defined(name) || !parms_[name].valid()) return false;
  Parameters parms(parms_);
  parms[name] = ""; // set illegal to avoid infinite recursion
  bool can = Expression<T>(parms_[name]).can_evaluate(ParameterEvaluator<T>(parms,this->evaluate_random()),isarg);
  return can;
}

template<class T>
Expression<T> ParameterEvaluator<T>::partial_evaluate(const std::string& name, bool isarg) const
{
  Expression<T> e;
  if (ParameterEvaluator<T>::can_evaluate(name,isarg))
    e=ParameterEvaluator<T>::evaluate(name,isarg);
  else if(!parms_.defined(name))
    e=Expression<T>(name);
  else {
    Parameters p(parms_);
    p[name]="";
    e=Expression<T>(static_cast<std::string>(parms_[name]));
    e.partial_evaluate(ParameterEvaluator<T>(p,this->evaluate_random()),isarg);
  }
  return e;
}

template<class T>
typename ParameterEvaluator<T>::value_type ParameterEvaluator<T>::evaluate(const std::string& name, bool isarg) const
{
  if (evaluate_helper<T>::can_evaluate_symbol(name,isarg))
    return evaluate_helper<T>::evaluate_symbol(name,isarg);
  if (parms_[name].template get<std::string>()=="Infinite recursion check" )
    boost::throw_exception(std::runtime_error("Infinite recursion when evaluating " + name));
  Parameters parms(parms_);
  parms[name] = "Infinite recursion check";
 typename ParameterEvaluator<T>::value_type res = alps::evaluate<value_type>(parms_[name], ParameterEvaluator<T>(parms,this->evaluate_random()), isarg);
  return res;
}

}
}

#endif // ! ALPS_EXPRESSION_IMPL_H
