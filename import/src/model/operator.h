/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/expression.h>
#include <alps/parameter.h>

namespace alps {

template <class T=std::complex<double> >
class OperatorEvaluator : public expression::ParameterEvaluator<T>
{
public:
  typedef expression::ParameterEvaluator<T> super_type;
  typedef typename super_type::value_type value_type;
  
  OperatorEvaluator(const Parameters& p)
    : super_type(p) {}
  typename super_type::Direction direction() const { return super_type::right_to_left; }

  value_type evaluate(const std::string& name, bool isarg=false) const
  { return super_type::partial_evaluate(name,isarg).value();}

  value_type evaluate_function(const std::string& name, const expression::Expression<T>& arg,bool isarg=false) const
  { return super_type::partial_evaluate_function(name,arg,isarg).value();}

  value_type evaluate_function(const std::string& name, const std::vector<expression::Expression<T> >& args,bool isarg=false) const
  { return super_type::partial_evaluate_function(name,args,isarg).value();}
};

} // namespace alps

#endif
