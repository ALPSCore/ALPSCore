/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2010 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_EXPRESSION_NUMBER_H
#define ALPS_EXPRESSION_NUMBER_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/evaluate_helper.h>
#include <alps/type_traits/real_type.hpp>
#include <boost/call_traits.hpp>

namespace alps {
namespace expression {

template<class T>
class Number : public Evaluatable<T> {
public:
  typedef T value_type;
  typedef typename alps::real_type<T>::type real_type;

  Number(typename boost::call_traits<value_type>::param_type x) : val_(x) {}
  value_type value(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  bool can_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false) const { return true; }
  void output(std::ostream&) const;
  Evaluatable<T>* clone() const { return new Number<T>(*this); }
private:
  value_type val_;
};

template<class T>
typename Number<T>::value_type Number<T>::value(const Evaluator<T>&, bool) const
{
  return val_;
}

template<class T>
void Number<T>::output(std::ostream& os) const
{
  if (evaluate_helper<T>::imag(val_) == 0)
    os << evaluate_helper<T>::real(val_);
  else
    os << val_;
}

} // end namespace expression
} // end namespace alps

#endif // ! ALPS_EXPRESSION_IMPL_H
