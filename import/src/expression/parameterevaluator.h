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
