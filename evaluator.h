/***************************************************************************
* ALPS library
*
* alps/evaluator.h   A class to evaluate expressions
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@comp-phys.org>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#ifndef ALPS_EVALUATOR_H
#define ALPS_EVALUATOR_H

#include <alps/parameters.h>
#include <string>

namespace alps {

class Expression;
class Evaluator {
public:
  enum Direction { left_to_right, right_to_left};
  Evaluator() {}
  virtual bool can_evaluate(const std::string&) const;
  virtual bool can_evaluate_function(const std::string&, const Expression& ) const;
  virtual double evaluate(const std::string&) const;
  virtual double evaluate_function(const std::string&, const Expression&) const;
  virtual Expression partial_evaluate(const std::string& name) const;
  virtual Expression partial_evaluate_function(const std::string& name, const Expression&) const;
  virtual Direction direction() const;
};

class ParameterEvaluator : public Evaluator {
public:
  ParameterEvaluator(const Parameters& p) : parms_(p) {}
  virtual ~ParameterEvaluator() {}
  bool can_evaluate(const std::string&) const;
  double evaluate(const std::string&) const;
  Expression partial_evaluate(const std::string& name) const;
  const Parameters& parameters() const { return parms_;}
private:
  Parameters parms_;
};
  
} // end namespace alps

#endif // ALPS_EVALUATOR_H
