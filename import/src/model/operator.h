/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
