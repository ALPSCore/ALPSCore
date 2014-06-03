/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_EXPRESSION_BLOCK_H
#define ALPS_EXPRESSION_BLOCK_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/evaluatable.h>

namespace alps {
namespace expression {

template<class T>
class Block : public Expression<T> {
private:
  typedef Expression<T> BASE_;

public:
  Block(std::istream&);
  Block(const Expression<T>& e) : BASE_(e) {}
  void output(std::ostream&) const;
  Evaluatable<T>* clone() const { return new Block<T>(*this); }
  void flatten();
  boost::shared_ptr<Evaluatable<T> > flatten_one();
  Evaluatable<T>* partial_evaluate_replace(const Evaluator<T>& =Evaluator<T>(),bool=false);
};

//
// implementation of Block<T>
//

template<class T>
Block<T>::Block(std::istream& in) : Expression<T>(in)
{
  char c;
  in >> c;
  if (c != ')' && c != ',')
    boost::throw_exception(std::runtime_error(") or , expected in expression"));
  if (c == ',') {
    // read imaginary part
    Expression<T> ex(in);
    Block<T> bl(ex);
    Term<T> term(bl);
    term *= "I";
    *this += term;
    check_character(in,')',") expected in expression");
  }
}

template<class T>
boost::shared_ptr<Evaluatable<T> > Block<T>::flatten_one()
{
  boost::shared_ptr<Expression<T> > ex = BASE_::flatten_one_expression();
  if (ex)
    return boost::shared_ptr<Evaluatable<T> >(new Block<T>(*ex));
  else
    return boost::shared_ptr<Evaluatable<T> >();
}

template<class T>
void Block<T>::output(std::ostream& os) const
{
  os << "(";
  BASE_::output(os);
  os << ")";
}

template<class T>
Evaluatable<T>* Block<T>::partial_evaluate_replace(const Evaluator<T>& p, bool isarg)
{
  Expression<T>::partial_evaluate(p,isarg);
  return this;
}

} // end namespace expression
} // end namespace alps

#endif // ! ALPS_EXPRESSION_IMPL_H
