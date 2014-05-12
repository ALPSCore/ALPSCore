/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_EXPRESSION_EVALUATE_H
#define ALPS_EXPRESSION_EVALUATE_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/expression.h>
#include <alps/expression/evaluator.h>
#include <alps/expression/evaluate_helper.h>
#include <alps/expression/traits.h>

namespace alps {

template<class T>
inline bool can_evaluate(const expression::Evaluatable<T>& ex, const expression::Evaluator<T>& ev, bool isarg=false)
{
  return ex.can_evaluate(ev,isarg);
}

template<class T>
inline bool can_evaluate(const std::string& v, const expression::Evaluator<T>& p, bool isarg=false)
{
  return expression::Expression<T>(v).can_evaluate(p,isarg);
}

inline bool can_evaluate(const std::string& v, const Parameters& p=Parameters())
{
  return can_evaluate(v, expression::ParameterEvaluator<>(p));
}

template<class U>
inline bool can_evaluate(const std::string& v, const Parameters& p, const U&)
{
  return can_evaluate(v, expression::ParameterEvaluator<U>(p));
}

inline bool can_evaluate(const StringValue& v, const Parameters& p=Parameters())
{
  return can_evaluate(static_cast<std::string>(v), p);
}

template<class U>
inline bool can_evaluate(const StringValue& v, const Parameters& p, const U&)
{
  return can_evaluate(static_cast<std::string>(v), p, U());
}

template<class U, class T>
inline U evaluate(const expression::Expression<T>& ex, const expression::Evaluator<T>& ev = expression::Evaluator<T>(), bool isarg=false)
{
  return expression::evaluate_helper<U>::value(ex, ev, isarg);
}

template<class U, class T>
inline U evaluate(const expression::Term<T>& ex, const expression::Evaluator<T>& ev = expression::Evaluator<T>(), bool isarg=false)
{
  return expression::evaluate_helper<U>::value(ex, ev,isarg);
}

template<class U, class T>
inline U evaluate(const char* v, const expression::Evaluator<T>& ev, bool isarg=false)
{
  return expression::evaluate_helper<U>::value(expression::Expression<T>(std::string(v)), ev,isarg);
}

template<class U, class T>
inline U evaluate(const std::string& v, const expression::Evaluator<T>& ev, bool isarg=false)
{
  return expression::evaluate_helper<U>::value(expression::Expression<T>(v), ev,isarg);
}

template<class U, class T>
inline U evaluate(const StringValue& v, const expression::Evaluator<T>& ev, bool isarg=false)
{
  return evaluate<U>(static_cast<std::string>(v), ev,isarg);
}

template<class U>
inline U evaluate(const char* v)
{
  return evaluate<U, U>(v, expression::Evaluator<
    typename expression::evaluate_helper<U>::value_type>());
}
inline double evaluate(const char* v) {
  return evaluate<double, double>(v, expression::Evaluator<
    expression::evaluate_helper<double>::value_type>());
}

template<class U>
inline U evaluate(const std::string& v)
{
  return evaluate<U, U>(v, expression::Evaluator<
    typename expression::evaluate_helper<U>::value_type>());
}
inline double evaluate(const std::string& v) {
  return evaluate<double, double>(v, expression::Evaluator<
    expression::evaluate_helper<double>::value_type>());
}

template<class U>
inline U evaluate(const StringValue& v)
{
  return evaluate<U, U>(v, expression::Evaluator<
    typename expression::evaluate_helper<U>::value_type>());
}
inline double evaluate(const StringValue& v) {
  return evaluate<double, double>(v, expression::Evaluator<
    expression::evaluate_helper<double>::value_type>());
}

template<class U>
inline U evaluate(const char* v, const Parameters& p)
{
  return evaluate<U, typename expression::evaluate_helper<U>::value_type>(v,
    expression::ParameterEvaluator<
    typename expression::evaluate_helper<U>::value_type>(p));
}
inline double evaluate(const char* v, const Parameters& p)
{
  return evaluate<double, expression::evaluate_helper<double>::value_type>(v,
    expression::ParameterEvaluator<
    expression::evaluate_helper<double>::value_type>(p));
}

template<class U>
inline U evaluate(const std::string& v, const Parameters& p)
{
  return evaluate<U,typename expression::evaluate_helper<U>::value_type>(v,
    expression::ParameterEvaluator<
    typename expression::evaluate_helper<U>::value_type>(p));
}
inline double evaluate(const std::string& v, const Parameters& p)
{
  return evaluate<double, expression::evaluate_helper<double>::value_type>(v,
    expression::ParameterEvaluator<
    expression::evaluate_helper<double>::value_type>(p));
}

template<class U>
inline U evaluate(const StringValue& v, const Parameters& p)
{
  return evaluate<U, typename expression::evaluate_helper<U>::value_type>(v,
    expression::ParameterEvaluator<
    typename expression::evaluate_helper<U>::value_type>(p));
}
inline double evaluate(const StringValue& v, const Parameters& p)
{
  return evaluate<double, expression::evaluate_helper<double>::value_type>(v,
    expression::ParameterEvaluator<
    expression::evaluate_helper<double>::value_type>(p));
}

template<class T>
void simplify(T) {}

template<class T>
void simplify(expression::Expression<T>& x) { x.simplify();}


StringValue simplify_value(StringValue const& val, Parameters const& parms, bool eval_random=false);

bool same_values(StringValue const& x, StringValue const& y, double eps=1e-15);

} // end namespace alps

#endif // ! ALPS_EXPRESSION_H
