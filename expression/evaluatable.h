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

#ifndef ALPS_EXPRESSION_EVALUATABLE_H
#define ALPS_EXPRESSION_EVALUATABLE_H

#include <alps/expression/expression_fwd.h>

namespace alps {
namespace expression {


template<class T>
class Evaluatable {
public:
  typedef T value_type;

  Evaluatable() {}
  virtual ~Evaluatable() {}
  virtual value_type value(const Evaluator<T>& =Evaluator<T>(), bool=false) const = 0;
  virtual bool can_evaluate(const Evaluator<T>& =Evaluator<T>(), bool=false) const = 0;
  virtual void output(std::ostream&) const = 0;
  virtual Evaluatable* clone() const = 0;
  virtual boost::shared_ptr<Evaluatable> flatten_one() { return boost::shared_ptr<Evaluatable>(); }
  virtual Evaluatable* partial_evaluate_replace(const Evaluator<T>& =Evaluator<T>(),bool=false) { return this; }
  virtual bool is_single_term() const { return false; }
  virtual Term<T> term() const;
  virtual bool depends_on(const std::string&) const { return false; }
};


template<class T>
inline Term<T> Evaluatable<T>::term() const { return Term<T>(); }

} // end namespace expression
} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
namespace expression {
#endif

template<class T>
inline std::ostream& operator<<(std::ostream& os, const alps::expression::Evaluatable<T>& e)
{
  e.output(os);
  return os;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace expression
} // end namespace alps
#endif

#endif // ! ALPS_EXPRESSION_IMPL_H
