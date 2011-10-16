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

#ifndef ALPS_EXPRESSION_EVALUATE_HELPER_H
#define ALPS_EXPRESSION_EVALUATE_HELPER_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/traits.h>
#include <alps/numeric/is_nonzero.hpp>

namespace alps {
namespace expression {

template<typename U, typename T>
struct numeric_cast_helper {
  static U value(typename boost::call_traits<T>::param_type x)
  {
    return x;
  }
};

template<typename U, typename T>
struct numeric_cast_helper<U, std::complex<T> > {
  static U value(const std::complex<T>& x) {
    if (numeric::is_nonzero(x.imag()))
      boost::throw_exception(std::runtime_error("can not convert complex number into real one"));
    return x.real();
  }
};

#ifndef BOOST_NO_SFINAE
template<typename U, typename T>
U numeric_cast(T x, typename boost::enable_if<boost::is_arithmetic<T> >::type* = 0)
{
  return numeric_cast_helper<U,T>::value(x);
}
template<typename U, typename T>
U numeric_cast(const T& x, typename boost::disable_if<boost::is_arithmetic<T> >::type* = 0)
{
  return numeric_cast_helper<U,T>::value(x);
}
#else
template<typename U, typename T>
U numeric_cast(const T& x)
{
  return numeric_cast_helper<U,T>::value(x);
}
#endif

template<class U>
struct evaluate_helper
{
  typedef U value_type;
  template<class R>
  static U value(const Term<R>& ex, const Evaluator<R>& =Evaluator<R>(), bool=false) { return ex; }
  template<class R>
  static U value(const Expression<R>& ex, const Evaluator<R>& =Evaluator<R>(), bool=false) { return ex; }
  static U real(typename boost::call_traits<U>::param_type u) { return u; }
};

template<class U>
struct evaluate_helper<Expression<U> >
{
  typedef U value_type;
  static Expression<U> value(const Term<U>& ex, const Evaluator<U>& ev=Evaluator<U>(), bool isarg=false) {
    Term<U> t(ex);
    t.partial_evaluate(ev,isarg);
    return t;
  }
  static Expression<U> value(const Expression<U>& ex, const Evaluator<U>& ev=Evaluator<U>(), bool isarg=false) {
    Expression<U> e(ex);
    e.partial_evaluate(ev,isarg);
    return e;
  }
};

template<>
struct evaluate_helper<double>
{
  typedef double value_type;
  template<class R>
  static double value(const Term<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return alps::expression::numeric_cast<double>(ex.value(ev,isarg));
  }
  template<class R>
  static double value(const Expression<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return alps::expression::numeric_cast<double>(ex.value(ev,isarg));
  }
  static double real(double u) { return u; }
  static double imag(double) { return 0; }
  static bool can_evaluate_symbol(const std::string& name,bool=false)
  {
    return (name=="Pi" || name=="PI" || name == "pi");
  }
  static double evaluate_symbol(const std::string& name,bool=false)
  {
    if (name=="Pi" || name=="PI" || name == "pi") return std::acos(-1.);
    boost::throw_exception(std::runtime_error("can not evaluate " + name));
    return 0.;
  }
};

template<>
struct evaluate_helper<float>
{
  typedef float value_type;
  template<class R>
  static float value(const Term<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return numeric_cast<float>(ex.value(ev,isarg));
  }
  template<class R>
  static float value(const Expression<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return numeric_cast<float>(ex.value(ev,isarg));
  }
  static float real(float u) { return u; }
  static float imag(float) { return 0; }
  static bool can_evaluate_symbol(const std::string& name,bool=false)
  {
    return (name=="Pi" || name=="PI" || name == "pi");
  }
  static float evaluate_symbol(const std::string& name,bool=false)
  {
    if (name=="Pi" || name=="PI" || name == "pi") return static_cast<float>(std::acos(-1.));
    boost::throw_exception(std::runtime_error("can not evaluate " + name));
    return 0.;
  }
};

template<>
struct evaluate_helper<long double>
{
  typedef long double value_type;
  template<class R>
  static long double value(const Term<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return numeric_cast<long double>(ex.value(ev,isarg));
  }
  template<class R>
  static long double value(const Expression<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return numeric_cast<long double>(ex.value(ev,isarg));
  }
  static long double real(long double u) { return u; }
  static long double imag(long double) { return 0; }
  static bool can_evaluate_symbol(const std::string& name,bool=false)
  {
    return (name=="Pi" || name=="PI" || name == "pi");
  }
  static long double evaluate_symbol(const std::string& name,bool=false)
  {
    if (name=="Pi" || name=="PI" || name == "pi") return std::acos(-1.);
    boost::throw_exception(std::runtime_error("can not evaluate " + name));
    return 0.;
  }
};

template<class U>
struct evaluate_helper<std::complex<U> >
{
  typedef std::complex<U> value_type;
  template<class R>
  static std::complex<U> value(const Term<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return ex.value(ev,isarg);
  }
  template<class R>
  static std::complex<U> value(const Expression<R>& ex, const Evaluator<R>& ev=Evaluator<R>(), bool isarg=false)
  {
    return ex.value(ev,isarg);
  }
  static U real(const std::complex<U>& u) { return u.real(); }
  static U imag(const std::complex<U>& u) { return u.imag(); }
  static bool can_evaluate_symbol(const std::string& name,bool=false)
  {
    return (name=="Pi" || name=="PI" || name == "pi" || name == "I");
  }
  static std::complex<U> evaluate_symbol(const std::string& name,bool=false)
  {
    if (name=="Pi" || name=="PI" || name == "pi") return std::acos(-1.);
    if (name=="I") return std::complex<U>(0.,1.);
    boost::throw_exception(std::runtime_error("can not evaluate " + name));
    return 0.;
  }
};

} // end namespace expression
} // end namespace alps


#endif // ! ALPS_EXPRESSION_H
