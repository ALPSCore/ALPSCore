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
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
