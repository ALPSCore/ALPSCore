/***************************************************************************
* ALPS library
*
* alps/expression_impl.h   A Class to evaluate expressions
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

#ifndef ALPS_EXPRESSION_IMPL_H
#define ALPS_EXPRESSION_IMPL_H

#include <alps/config.h>
#include <alps/expression.h>

#include <iostream>

namespace alps {
namespace detail {

class Block : public Expression
{
public:
  Block(std::istream&);
  Block(const Expression& e) : Expression(e) {}
  void output(std::ostream&) const;
  Evaluatable* clone() const;
  void flatten();
  boost::shared_ptr<Evaluatable> flatten_one();
  Evaluatable* partial_evaluate_replace(const Evaluator& p);
};

class Symbol : public Evaluatable {
public:
  Symbol(const std::string& n) : name_(n) {}
  double value(const Evaluator& p) const;
  bool can_evaluate(const Evaluator& p) const;
  void output(std::ostream&) const;
  Evaluatable* clone() const;
  Evaluatable* partial_evaluate_replace(const Evaluator& p);
  bool depends_on(const std::string& s) const;
private:
  std::string name_;
};

class Function : public Evaluatable {
public:
  Function(std::istream&, const std::string&);
  Function(const std::string& n, const Expression& e) : name_(n), arg_(e) {}
  double value(const Evaluator& p) const;
  bool can_evaluate(const Evaluator& p) const;
  void output(std::ostream&) const;
  Evaluatable* clone() const;
  boost::shared_ptr<Evaluatable> flatten_one();
  Evaluatable* partial_evaluate_replace(const Evaluator& p);
  bool depends_on(const std::string& s) const;
private:
 std::string name_;
 Expression arg_;
};

class Number : public Evaluatable {
public:
  Number(double x) : val_(x) {}
  double value(const Evaluator& p) const;
  bool can_evaluate(const Evaluator& p) const;
  void output(std::ostream&) const;
  Evaluatable* clone() const;
private:
 double val_;
};

} // end namespace detail
} // end namespace alps

#endif // ALPS_EXPRESSION_IMPL_H
