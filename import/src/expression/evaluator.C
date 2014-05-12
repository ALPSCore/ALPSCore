/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/expression/evaluator.h>
#include <alps/expression/evaluate.h>
#include <alps/numeric/is_zero.hpp>

namespace alps {

Disorder::random_type Disorder::rng_;

int Disorder::last_seed_;
  
ALPS_DECL  boost::variate_generator<Disorder::random_type&, boost::uniform_real<> > 
  Disorder::random(Disorder::rng_, boost::uniform_real<>());
    
ALPS_DECL  boost::variate_generator<Disorder::random_type&, boost::normal_distribution<> > 
  Disorder::gaussian_random(Disorder::rng_, boost::normal_distribution<>());

void Disorder::seed(unsigned int i) 
{ 
  seed_with_sequence(rng_,i);
  last_seed_=i;
}

void Disorder::seed_if_unseeded(const alps::Parameters& p) 
{
  int s = p.value_or_default("DISORDERSEED",0);
  if (s && s != last_seed_)
    seed(s);
}

StringValue simplify_value(StringValue const& val, Parameters const& parms, bool eval_random)
{
  try {
    expression::ParameterEvaluator<double> eval(parms,eval_random);
    expression::Expression<double> expr(val);
    if (expr.can_evaluate(eval)) {
        double value = expr.value(eval);
        if (numeric::is_zero(value - static_cast<double>(static_cast<int>(value)))) 
            return static_cast<int>(value);
        else 
            return value;
    } else {
        expr.partial_evaluate(eval);
        return boost::lexical_cast<std::string>(expr);
    }
  } catch(...) {
    // we had a problem evaluating, use original full value
    return val;
  }
}

bool same_values(StringValue const& x, StringValue const& y, double eps)
{
  try {
    //expression::ParameterEvaluator<double> eval();
    expression::Expression<double> exprx(x);
    expression::Expression<double> expry(y);
    expression::ParameterEvaluator<double> eval(Parameters(),false);
    if (exprx.can_evaluate(eval) && expry.can_evaluate(eval)) {
      return std::abs(exprx.value()-expry.value()) <= eps*std::max(std::abs(exprx.value()),std::abs(expry.value())) ;
    } else {
      return x==y;
    }
  } catch(...) {
    // we had a problem evaluating, use original full value
    return x==y;
  }
}

} // end namespace alps
