/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_EXPRESSION_TRAITS_H
#define ALPS_EXPRESSION_TRAITS_H

#include <alps/expression/expression_fwd.h>

namespace alps {
namespace expression {

template<class T>
struct expression {
  typedef std::complex<T> value_type;
  typedef Expression<value_type> type;
  typedef Term<value_type> term_type;
};

template<class T>
struct expression<std::complex<T> > {
  typedef std::complex<T> value_type;
  typedef Expression<value_type> type;
  typedef Term<value_type> term_type;
};

template<class T>
struct expression<Expression<T> > {
  typedef T value_type;
  typedef Expression<value_type> type;
  typedef Term<value_type> term_type;
};

}


template <class T>
struct expression_value_type_traits {
  typedef typename expression::expression<T>::value_type value_type;
};

//
// function is_zero and is_nonzero
//

template<class T>
bool is_zero(const expression::Expression<T>& x)
{
  std::string s = boost::lexical_cast<std::string>(x);
  return s=="" || s=="0" || s=="0." || s=="-0" || s=="-0.";
}

template<class T>
bool is_zero(const expression::Term<T>& x)
{
  std::string s = boost::lexical_cast<std::string>(x);
  return s=="" || s=="0" || s=="0.";
}


} // end namespace alps

#endif // ! ALPS_EXPRESSION_H
