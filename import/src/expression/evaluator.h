/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_EXPRESSION_EVALUATOR_H
#define ALPS_EXPRESSION_EVALUATOR_H

#include <alps/expression/expression_fwd.h>
#include <alps/expression/evaluate_helper.h>
#include <alps/config.h>

namespace alps {

class ALPS_DECL Disorder
{
public:
  typedef boost::mt19937 random_type;
private:
  static random_type rng_;
  static int last_seed_;
public:
  static boost::variate_generator<random_type&,boost::uniform_real<> > random;
  static boost::variate_generator<random_type&,boost::normal_distribution<> > gaussian_random;
  static void seed(unsigned int =0);
  static void seed_if_unseeded(const alps::Parameters&);
};

namespace expression {

template<class T>
class Evaluator {
public:
  typedef T value_type;
  enum Direction { left_to_right, right_to_left };
  Evaluator(bool rand=true) : evaluate_random_(rand) {}
  virtual ~Evaluator() {}

  virtual bool can_evaluate(const std::string&, bool=false) const;
  virtual bool can_evaluate_function(const std::string&, const Expression<T>&, bool=false) const;
  virtual bool can_evaluate_function(const std::string&, const std::vector<Expression<T> >&, bool=false) const;
  virtual value_type evaluate(const std::string&, bool=false) const;
  virtual value_type evaluate_function(const std::string&, const Expression<T>&, bool=false) const;
  virtual value_type evaluate_function(const std::string&, const std::vector<Expression<T> >&, bool=false) const;
  virtual Expression<T> partial_evaluate(const std::string& name, bool=false) const;
  virtual Expression<T> partial_evaluate_function(const std::string& name, const Expression<T>&, bool=false) const;
  virtual Expression<T> partial_evaluate_function(const std::string& name, const std::vector<Expression<T> >&, bool=false) const;
  virtual Direction direction() const;

  bool can_evaluate_expressions(const std::vector<Expression<T> >&, bool=false) const;
  void partial_evaluate_expressions(std::vector<Expression<T> >&, bool=false) const;
  bool evaluate_random() const { return evaluate_random_;}

private:
  bool evaluate_random_;
};


template<class T>
class ParameterEvaluator : public Evaluator<T> {
public:
  typedef Evaluator<T> super_type;
  typedef T value_type;
  ParameterEvaluator(const Parameters& p, bool rand=true) 
  : Evaluator<T>(rand), parms_(p) { Disorder::seed_if_unseeded(p);}
  virtual ~ParameterEvaluator() {}

  bool can_evaluate(const std::string&, bool=false) const;
  value_type evaluate(const std::string&, bool=false) const;
  Expression<T> partial_evaluate(const std::string& name, bool=false) const;
  const Parameters& parameters() const { return parms_;}
protected:
  void set_parameters(const Parameters& p) { parms_=p;}
private:
  Parameters parms_;
};

template<class T>
bool Evaluator<T>::can_evaluate(const std::string&, bool) const
{
  return false;
}

template<class T>
bool Evaluator<T>::can_evaluate_function(const std::string& name, const Expression<T>& arg, bool) const
{
  return arg.can_evaluate(*this,true) &&
         (name=="sqrt" || name=="abs" ||
          name=="sin" || name=="cos" || name=="tan" ||
          name=="asin" || name=="acos" || name=="atan" ||
          name=="log" || name=="exp" || (evaluate_random_ && name=="integer_random"));
}


template<class T>
bool Evaluator<T>::can_evaluate_expressions(const std::vector<Expression<T> >& arg, bool f) const
{
  bool can=true;
  for (typename std::vector<Expression<T> >::const_iterator it=arg.begin();it!=arg.end();++it)
    can = can && it->can_evaluate(*this,f);
  return can;
}

template<class T>
void Evaluator<T>::partial_evaluate_expressions(std::vector<Expression<T> >& arg, bool f) const
{
  for (typename std::vector<Expression<T> >::iterator it=arg.begin();it!=arg.end();++it) {
   it->partial_evaluate(*this,f);
   it->simplify();
  }
}


template<class T>
bool Evaluator<T>::can_evaluate_function(const std::string& name, const std::vector<Expression<T> >& arg, bool f) const
{
  bool can= can_evaluate_expressions(arg,true) &&
       ((arg.size()==0 && evaluate_random_ && (name == "random" || name=="gaussian_random" || name == "normal_random")) ||
        (arg.size()==1 && can_evaluate_function(name,arg[0],f)) || 
        (arg.size()==2 && (evaluate_random_ && (name=="gaussian_random" || name=="atan2"))));
  return can;
}


template<class T>
typename Evaluator<T>::Direction Evaluator<T>::direction() const
{
  return Evaluator<T>::left_to_right;
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate(const std::string& name,bool isarg) const
{
  return partial_evaluate(name,isarg).value();
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate_function(const std::string& name, const Expression<T>& arg,bool isarg) const
{
  return partial_evaluate_function(name,arg,isarg).value();
}

template<class T>
typename Evaluator<T>::value_type Evaluator<T>::evaluate_function(const std::string& name, const std::vector<Expression<T> >& arg,bool isarg) const
{
  return partial_evaluate_function(name,arg,isarg).value();
}

template<class T>
Expression<T> Evaluator<T>::partial_evaluate(const std::string& name,bool) const
{
  return Expression<T>(name);
}


template<class T>
Expression<T> Evaluator<T>::partial_evaluate_function(const std::string& name, const Expression<T>& arg,bool) const
{
  if(!arg.can_evaluate(*this,true)) {
    Expression<T> e(arg);
    e.partial_evaluate(*this,true);
    return Expression<T>(Function<T>(name,e));
  }
  value_type val=arg.value(*this,true);
  if (name=="sqrt")
    val = std::sqrt(val);
  else if (name=="abs")
    val = std::abs(val);
  else if (name=="sin")
    val = std::sin(val);
  else if (name=="cos")
    val = std::cos(val);
  else if (name=="tan")
    val = std::tan(val);
  else if (name=="asin")
    val = std::asin(evaluate_helper<T>::real(val));
  else if (name=="acos")
    val = std::acos(evaluate_helper<T>::real(val));
  else if (name=="atan")
    val = std::atan(evaluate_helper<T>::real(val));
  else if (name=="exp")
    val = std::exp(val);
  else if (name=="log")
    val = std::log(val);
  else if (name=="integer_random" && evaluate_random_)
    val=static_cast<int>(evaluate_helper<T>::real(val)*Disorder::random());
  else
    return Expression<T>(Function<T>(name,Expression<T>(val)));
  return Expression<T>(val);
}

template<class T>
Expression<T> Evaluator<T>::partial_evaluate_function(const std::string& name, const std::vector<Expression<T> >& args,bool isarg) const
{
  if (args.size()==1)
    return partial_evaluate_function(name,args[0],isarg);
    
  std::vector<Expression<T> > evaluated;
  bool could_evaluate = true;
  for (typename std::vector<Expression<T> >::const_iterator it = args.begin(); it !=args.end();++it) {
    evaluated.push_back(*it);
    could_evaluate = could_evaluate && it->can_evaluate(*this,true);
    evaluated.rbegin()->partial_evaluate(*this,true);
  }
  if (evaluated.size()==2 && could_evaluate) {
    double arg1=evaluate_helper<T>::real(evaluated[0].value());
    double arg2=evaluate_helper<T>::real(evaluated[1].value());
    if (name=="atan2")
      return Expression<T>(static_cast<T>(std::atan2(arg1,arg2)));
    else if (evaluate_random_ && (name=="gaussian_random" || name=="normal_random"))
      return Expression<T>(arg1+arg2*Disorder::gaussian_random());
  }
  else if (evaluated.size()==0) {
    if (evaluate_random_ && name=="random")
      return Expression<T>(Disorder::random());
    else if (evaluate_random_ && (name=="gaussian_random" || name=="normal_random"))
      return Expression<T>(Disorder::gaussian_random());
  }
  return Expression<T>(Function<T>(name,evaluated));
}

}
}

#endif // ! ALPS_EXPRESSION_EVALUATOR_H
