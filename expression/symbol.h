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

#ifndef ALPS_EXPRESSION_SYMBOL_H
#define ALPS_EXPRESSION_SYMBOL_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/evaluatable.h>

namespace alps {
namespace expression {

template<class T>
class Symbol : public Evaluatable<T> {
public:
  typedef T value_type;

  Symbol(const std::string& n) : name_(n) {}
  value_type value(const Evaluator<T>& =Evaluator<T>(), bool=false) const;
  bool can_evaluate(const Evaluator<T>& ev=Evaluator<T>(), bool isarg=false) const
  { return ev.can_evaluate(name_,isarg);}
  void output(std::ostream& os) const { os << name_; }
  Evaluatable<T>* clone() const { return new Symbol<T>(*this); }
  Evaluatable<T>* partial_evaluate_replace(const Evaluator<T>& =Evaluator<T>(), bool=false);
  bool depends_on(const std::string& s) const;
private:
  std::string name_;
};

template<class T>
bool Symbol<T>::depends_on(const std::string& s) const {
  return (name_==s);
}

template<class T>
typename Symbol<T>::value_type Symbol<T>::value(const Evaluator<T>& eval, bool isarg) const
{
  if (!eval.can_evaluate(name_,isarg))
    boost::throw_exception(std::runtime_error("Cannot evaluate " + name_ ));
  return eval.evaluate(name_,isarg);
}

template<class T>
Evaluatable<T>* Symbol<T>::partial_evaluate_replace(const Evaluator<T>& p, bool isarg)
{
  Expression<T> e(p.partial_evaluate(name_,isarg));
  if (e==name_)
    return this;
  else
    return new Block<T>(p.partial_evaluate(name_,isarg));
}

} // end namespace expression
} // end namespace alps

#endif // ! ALPS_EXPRESSION_IMPL_H
