/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
