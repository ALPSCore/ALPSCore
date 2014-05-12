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
