/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/evaluator.h>
#include <alps/expression_impl.h>
#include <alps/expression.h>

#include <boost/throw_exception.hpp>
#include <cmath>
#include <stdexcept>

namespace alps {

bool Evaluator::can_evaluate(const std::string&) const 
{ 
  return false;
}

bool Evaluator::can_evaluate_function(const std::string& name, const Expression& arg) const 
{ 
  return arg.can_evaluate(*this) &&
         (name=="sin" || name=="cos" || name=="tan" ||
          name=="log" || name=="exp" || name=="sqrt");
}

Evaluator::Direction Evaluator::direction() const 
{ 
  return Evaluator::left_to_right;
}

double Evaluator::evaluate(const std::string& name) const 
{ 
  return partial_evaluate(name).value();
}

double Evaluator::evaluate_function(const std::string& name, const Expression& arg) const 
{ 
  return partial_evaluate_function(name,arg).value();
}

Expression Evaluator::partial_evaluate(const std::string& name) const 
{ 
  return Expression(name);
}

Expression Evaluator::partial_evaluate_function(const std::string& name, const Expression& arg) const 
{ 
  if(!arg.can_evaluate(*this)) {
    Expression e(arg);
    e.partial_evaluate(*this);
    return Expression(detail::Function(name,e));
  }
  double val=arg.value(*this);
  if (name=="sqrt")
    val = std::sqrt(val);
  else if (name=="sin")
    val = std::sin(val);
  else if (name=="cos")
    val = std::cos(val);
  else if (name=="tan")
    val = std::tan(val);
  else if (name=="exp")
    val = std::exp(val);
  else if (name=="log")
    val = std::log(val);
  else
    return Expression(detail::Function(name,Expression(val)));
  return Expression(val);
}

bool ParameterEvaluator::can_evaluate(const std::string& name) const
{
  if (name=="Pi" || name=="PI" || name == "pi")
    return true;
  if (!parms_.defined(name) )
    return false;
  Parameters parms(parms_);
  parms[name]=""; // set illegal to avoid infinite recursion
  return (name=="Pi" || name=="PI" || name == "pi" ||
          alps::can_evaluate(parms_[name], ParameterEvaluator(parms)));
}

Expression ParameterEvaluator::partial_evaluate(const std::string& name) const
{
  if ( ParameterEvaluator::can_evaluate(name)) 
    return Expression(ParameterEvaluator::evaluate(name));
  if(!parms_.defined(name))
    return Expression(name);
  Parameters p(parms_);
  p[name]="";
  Expression e(static_cast<std::string>(parms_[name]));
  e.partial_evaluate(ParameterEvaluator(p));
  return e;
}

double ParameterEvaluator::evaluate(const std::string& name) const
{
  if (name=="Pi" || name == "PI" || name=="pi")
    return std::acos(-1.);
  if (parms_[name].get<std::string>()=="Infinite recursion check" )
    boost::throw_exception(std::runtime_error("Infinite recursion when evaluating " + name));
  Parameters parms(parms_);
  parms[name]="Infinite recursion check";
  ParameterEvaluator eval(parms);
  return ::alps::evaluate(parms_[name], eval);
}

} // namespace alps
