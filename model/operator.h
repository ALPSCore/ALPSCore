/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2004 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_OPERATOR_H
#define ALPS_MODEL_OPERATOR_H

#include <alps/model/operatordescriptor.h>
#include <alps/expression.h>
#include <alps/parameters.h>
#include <alps/evaluator.h>
#include <vector>

namespace alps {

template <class I>
class OperatorEvaluator : public ParameterEvaluator
{
public:
  typedef typename OperatorDescriptor<I>::operator_map operator_map;
  typedef typename operator_map::const_iterator operator_iterator;
  typedef ParameterEvaluator super_type;
  typedef super_type::value_type value_type;
  
  OperatorEvaluator(const Parameters& p, const operator_map& o)
    : ParameterEvaluator(p), ops_(o) {}
  Direction direction() const { return right_to_left; }

  bool has_operator(const std::string& name) const
  { return ops_.find(name) != ops_.end();}

  value_type evaluate(const std::string& name) const
  { return partial_evaluate(name).value();}

  value_type evaluate_function(const std::string& name, const Expression& arg) const
  { return partial_evaluate_function(name,arg).value();}
  
protected:
  const operator_map& ops_;
};

} // namespace alps

#endif
