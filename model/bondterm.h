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

#ifndef ALPS_MODEL_BONDTERM_H
#define ALPS_MODEL_BONDTERM_H

#include <alps/model/operator.h>
#include <alps/expression.h>
#include <alps/multi_array.hpp>
#include <alps/parameters.h>

namespace alps {

template<class I>
class BondTermDescriptor
{
public:
  typedef std::map<std::string,OperatorDescriptor<I> > operator_map;

  BondTermDescriptor() : type_(-2) {}
  BondTermDescriptor(const std::string& term)
    : type_(-2), term_(term), source_("i"), target_("j") {}
  BondTermDescriptor(const std::string& term, const std::string& source,
                     const std::string& target)
    : type_(-2), term_(term), source_(source), target_(target) {}
  BondTermDescriptor(const Term& term)
    : type_(-2), term_(boost::lexical_cast<std::string>(term)),
      source_("i"), target_("j") {}
  BondTermDescriptor(const Term& term, const std::string& source,
                     const std::string& target)
    : type_(-2), term_(boost::lexical_cast<std::string>(term)),
      source_(source), target_(target) {}
  BondTermDescriptor(const XMLTag&, std::istream&);
  void write_xml(oxstream&) const;

  bool match_type(int type) const { return type_==-1 || type==type_;}
  const std::string& term () const { return term_;}
  const std::string& source () const { return source_;}
  const std::string& target () const { return target_;}

  template <class T>
  boost::multi_array<std::pair<T,std::pair<bool,bool> >,4> matrix(const SiteBasisDescriptor<I>&, const SiteBasisDescriptor<I>&, const operator_map&, const Parameters& =Parameters()) const;
  std::set<Term> split(const operator_map&, const Parameters& = Parameters()) const;

private:
  int type_;
  std::string term_;
  std::string source_;
  std::string target_;
};


template <class I, class STATE1=site_state<I>, class STATE2=site_state<I>  >
class BondOperatorEvaluator : public OperatorEvaluator<I>
{
private:
  typedef OperatorEvaluator<I> super_type;
  typedef BondOperatorEvaluator<I, STATE1, STATE2> SELF_;

public:
  typedef typename super_type::operator_map operator_map;

  BondOperatorEvaluator(const STATE1& s1, const STATE2& s2,
                        const SiteBasisDescriptor<I>& b1,
			const SiteBasisDescriptor<I>& b2,
                        std::string site1, std::string site2,
			const Parameters& p, const operator_map& o)
    : super_type(p,o), state_(s1,s2), basis1_(b1), basis2_(b2),
      sites_(site1,site2), fermionic_(false,false) {}
  bool can_evaluate_function(const std::string& name,
			     const Expression& argument) const;
  Expression partial_evaluate_function(const std::string& name,
					const Expression& argument) const;
  const std::pair<STATE1,STATE2>& state() const { return state_;}
  std::pair<bool,bool> fermionic() const { return fermionic_;}

private:
  mutable std::pair<STATE1,STATE2> state_;
  const SiteBasisDescriptor<I>& basis1_;
  const SiteBasisDescriptor<I>& basis2_;
  std::pair<std::string, std::string> sites_;
  mutable std::pair<bool,bool> fermionic_;
};

template <class I>
class BondOperatorSplitter : public OperatorEvaluator<I>
{
private:
  typedef OperatorEvaluator<I> super_type;
  typedef BondOperatorSplitter<I> SELF_;

public:
  typedef typename super_type::operator_map operator_map;

  BondOperatorSplitter(std::string site1, std::string site2,
                       const Parameters& p, const operator_map& o)
    : super_type(p,o), sites_(site1,site2) {}

  bool can_evaluate_function(const std::string& name, const Expression& argument) const;
  Expression partial_evaluate_function(const std::string& name, const Expression& argument) const;
  const std::pair<Term, Term>& site_operators() const { return site_ops_; }

private:
  mutable std::pair<Term, Term> site_ops_;
  std::pair<std::string, std::string> sites_;
};

template <class I, class STATE1, class STATE2>
bool BondOperatorEvaluator<I, STATE1, STATE2>::can_evaluate_function(const std::string& name, const Expression& arg) const
{
  if (super_type::ops_.find(name) != super_type::ops_.end() && (arg== sites_.first || arg==sites_.second)) {
    SELF_ eval(*this);
    return eval.partial_evaluate_function(name,arg).can_evaluate(ParameterEvaluator(*this));
  }
  else
    return ParameterEvaluator::can_evaluate_function(name,arg);
}

template <class I>
bool BondOperatorSplitter<I>::can_evaluate_function(const std::string& name, const Expression& arg) const
{
  return (super_type::ops_.find(name) != super_type::ops_.end() 
          && (arg== sites_.first || arg==sites_.second)) ||
         ParameterEvaluator::can_evaluate_function(name,arg);
}

template <class I, class STATE1, class STATE2>
Expression BondOperatorEvaluator<I, STATE1, STATE2>::partial_evaluate_function(const std::string& name, const Expression& arg) const
{
  typename operator_map::const_iterator op = super_type::ops_.find(name);
  Expression e;
  if (op!=super_type::ops_.end()) {  // evaluate operator
    bool f;
    if (arg==sites_.first) {
      boost::tie(state_.first,e,f) = op->second.apply(state_.first,basis1_,*this);
      if (f && is_nonzero(e)) {
        fermionic_.first=!fermionic_.first;
        if (fermionic_.second) // for normal ordering
          e.negate(); 
      }
    }
    else  if (arg==sites_.second) {
      boost::tie(state_.second,e,f) = op->second.apply(state_.second,basis2_,*this);
      if (f)
        fermionic_.second=!fermionic_.second;
    }
    else {
      e=ParameterEvaluator(*this).partial_evaluate_function(name,arg);
}
  }
  else
    e=ParameterEvaluator(*this).partial_evaluate_function(name,arg);
  return e;
}

template <class I>
Expression BondOperatorSplitter<I>::partial_evaluate_function(const std::string& name, const Expression& arg) const
{
  typename operator_map::const_iterator op = super_type::ops_.find(name);
  if (op!=super_type::ops_.end()) {  // evaluate operator
    Expression e;
    if (arg==sites_.first)
      site_ops_.first *= name;
    else  if (arg==sites_.second)
      site_ops_.second *= name;
    else
      return ParameterEvaluator(*this).partial_evaluate_function(name,arg);
    return Expression(1.);
  }
  else
    return ParameterEvaluator(*this).partial_evaluate_function(name,arg);
}

template <class I, class T>
boost::multi_array<std::pair<T,std::pair<bool,bool> >,4> get_fermionic_matrix(T,const BondTermDescriptor<I>& m, const SiteBasisDescriptor<I>& basis1, const SiteBasisDescriptor<I>& basis2, const  typename BondTermDescriptor<I>::operator_map& ops, const Parameters& p=Parameters())
{
  return m.template matrix<T>(basis1,basis2,ops,p);
}

template <class I, class T>
boost::multi_array<T,4> get_matrix(T,const BondTermDescriptor<I>& m, const SiteBasisDescriptor<I>& basis1, const SiteBasisDescriptor<I>& basis2, const  typename BondTermDescriptor<I>::operator_map& ops, const Parameters& p=Parameters())
{
  boost::multi_array<std::pair<T,std::pair<bool,bool> >,4> f_matrix = m.template matrix<T>(basis1,basis2,ops,p);
  boost::multi_array<T,4> matrix(boost::extents[f_matrix.shape()[0]][f_matrix.shape()[1]][f_matrix.shape()[2]][f_matrix.shape()[3]]);
  for (int i=0;i<f_matrix.shape()[0];++i)
    for (int j=0;j<f_matrix.shape()[1];++j)
      for (int k=0;k<f_matrix.shape()[2];++k)
        for (int l=0;l<f_matrix.shape()[3];++l)
          if (f_matrix[i][j][k][l].second.first || f_matrix[i][j][k][l].second.second)
            boost::throw_exception(std::runtime_error("Cannot convert fermionic operator to a bosonic matrix"));
          else
           matrix[i][j][k][l]=f_matrix[i][j][k][l].first;
  return matrix;
}

template <class I> template <class T> boost::multi_array<std::pair<T,std::pair<bool,bool> >,4>
BondTermDescriptor<I>::matrix(const SiteBasisDescriptor<I>& b1,
                              const SiteBasisDescriptor<I>& b2,
                              const operator_map& ops,
                              const Parameters& p) const
{
  SiteBasisDescriptor<I> basis1(b1);
  SiteBasisDescriptor<I> basis2(b2);
  basis1.set_parameters(p);
  basis2.set_parameters(p);
  Parameters parms(p);
  parms.copy_undefined(basis1.get_parameters());
  parms.copy_undefined(basis2.get_parameters());
  std::size_t dim1=basis1.num_states();
  std::size_t dim2=basis2.num_states();
  boost::multi_array<std::pair<T,std::pair<bool,bool> >,4> mat(boost::extents[dim1][dim2][dim1][dim2]);
  // parse expression and store it as sum of terms
  Expression ex(term());
  ex.flatten();
  // fill the matrix
    site_basis<I> states1(basis1);
    site_basis<I> states2(basis2);
    for (int i=0;i<mat.shape()[0];++i)
      for (int j=0;j<mat.shape()[1];++j)
        for (int k=0;k<mat.shape()[2];++k)
          for (int l=0;l<mat.shape()[3];++l)
            mat[i][j][k][l].second=std::make_pair(false,false);
    for (int i1=0;i1<states1.size();++i1)
      for (int i2=0;i2<states2.size();++i2) {
      //calculate expression applied to state *it and store it into matrix
        for (typename Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
          BondOperatorEvaluator<I> evaluator(states1[i1], states2[i2], basis1, basis2, source(), target(), parms, ops);
          Term term(*tit);
          term.partial_evaluate(evaluator);
          unsigned int j1=states1.index(evaluator.state().first);
          unsigned int j2=states2.index(evaluator.state().second);
	      if (is_nonzero(term) && j1<dim1 && j2<dim2) {
            if (is_nonzero(mat[i1][i2][j1][j2].first)) {
              if (mat[i1][i2][j1][j2].second.first != evaluator.fermionic().first || 
                  mat[i1][i2][j1][j2].second.second != evaluator.fermionic().second) 
              boost::throw_exception(std::runtime_error("Inconsistent fermionic nature of a matrix element: "
                                    + boost::lexical_cast<std::string,Term>(*tit) + " is inconsistent with "
                                    + boost::lexical_cast<std::string,T>(mat[i1][i2][j1][j2].first) + 
                                    ". Please contact the library authors for an extension to the ALPS model library."));
            }
            else
              mat[i1][i2][j1][j2].second=evaluator.fermionic();
            if (boost::is_arithmetic<T>::value || TypeTraits<T>::is_complex)
              if (!can_evaluate(boost::lexical_cast<std::string>(term)))
                boost::throw_exception(std::runtime_error("Cannot evaluate expression " + boost::lexical_cast<std::string>(term)));
#ifndef ALPS_WITH_NEW_EXPRESSION
            mat[i1][i2][j1][j2].first += term;
#else
            mat[i1][i2][j1][j2].first += evaluate<T>(term);
#endif
        }
      }
    }
  return mat;
}

template <class I>
std::set<Term> BondTermDescriptor<I>::split(const operator_map& ops, const Parameters& p) const
{
  std::set<Term> terms;
  Expression ex(term());
  ex.flatten();
  for (typename Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
    BondOperatorSplitter<I> evaluator(source(),target(),p,ops);
    Term term(*tit);
    term.partial_evaluate(evaluator);
    terms.insert(evaluator.site_operators().first);
    terms.insert(evaluator.site_operators().second);
  }
  return terms;
}

#ifndef ALPS_WITHOUT_XML

template <class I>
BondTermDescriptor<I>::BondTermDescriptor(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  type_ = tag.attributes["type"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["type"]);
  source_ = tag.attributes["source"]=="" ? std::string("i") : tag.attributes["source"];
  target_ = tag.attributes["target"]=="" ? std::string("j") : tag.attributes["target"];
  if (tag.type!=XMLTag::SINGLE) {
    term_=parse_content(is);
    tag = parse_tag(is);
    if (tag.name !="/BONDTERM")
      boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <BONDTERM> element"));
  }
}

template <class I>
void BondTermDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("BONDTERM");
  if (type_>=0)
    os << attribute("type", type_);
  if (term()!="")
    os << attribute("source", source()) << attribute("target", target()) << term();
  os << end_tag("BONDTERM");
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::BondTermDescriptor<I>& q)
{
  q.write_xml(out);
  return out;
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::BondTermDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
