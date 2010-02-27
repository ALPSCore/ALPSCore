/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2010 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_BONDOPERATOR_H
#define ALPS_MODEL_BONDOPERATOR_H

#include <alps/config.h>
#include <alps/type_traits/is_complex.hpp>
#include <alps/model/operator.h>
#include <alps/model/siteoperator.h>
#include <alps/model/sitestate.h>
#include <alps/model/sitebasisstates.h>
#include <alps/expression.h>
#include <alps/parameter.h>
#include <alps/multi_array.hpp>

namespace alps {

template <class I, class T=std::complex<double>, class STATE1=site_state<I>, class STATE2=site_state<I> > class BondOperatorEvaluator;
class ModelLibrary;

template <class I, class T=std::complex<double> >
class BondOperatorSplitter : public OperatorEvaluator<T>
{
private:
  typedef OperatorEvaluator<T> super_type;
  typedef BondOperatorSplitter<I,T> SELF_;

public:
  BondOperatorSplitter(const SiteBasisDescriptor<I>& b1,
                       const SiteBasisDescriptor<I>& b2,
                       const std::string& site1, const std::string& site2,
                       const Parameters& p)
    : super_type(p), basis1_(b1), basis2_(b2), sites_(site1,site2), second_site_fermionic_(false) {}

  bool can_evaluate_function(const std::string& name, const expression::Expression<T>& argument,bool=false) const;
  expression::Expression<T> partial_evaluate_function(const std::string& name, const expression::Expression<T>& argument,bool=false) const;
  const std::pair<expression::Term<T>, expression::Term<T> >& site_operators() const { return site_ops_; }
  bool has_operator(const std::string& name, const expression::Expression<T>& arg) const
  {
    return (arg==sites_.first && basis1_.has_operator(name)) ||
           (arg==sites_.second && basis2_.has_operator(name));
  }

private:
  const SiteBasisDescriptor<I>& basis1_;
  const SiteBasisDescriptor<I>& basis2_;
  mutable std::pair<expression::Term<T>, expression::Term<T> > site_ops_;
  std::pair<std::string, std::string> sites_;
  mutable bool second_site_fermionic_;
};

class ALPS_DECL BondOperator
{
public:
  BondOperator() : source_("i"), target_("j") {}
  BondOperator(const std::string& s, const std::string& t) : source_(s), target_(t) {}
  //BondOperator(const BondOperator& op) : name_(op.name_), term_(op.term_), source_(op.source_), target_(op.target_) {}
  // template <class T>
  // BondOperator(const T& term)
  //   : term_(boost::lexical_cast<std::string>(term)), source_("i"), target_("j") {}
  // template <class T>
  // BondOperator(const T& term, const std::string& s)
  //   : term_(boost::lexical_cast<std::string>(term)), source_(s), target_("j") {}
  // template <class T>
  // BondOperator(const T& term, const std::string& s, const std::string& t)
  //   : term_(boost::lexical_cast<std::string>(term)), source_(s), target_(t) {}
  BondOperator(const std::string& term, const std::string& s, const std::string& t)
    : term_(term), source_(s), target_(t) {}
  BondOperator(const XMLTag& tag, std::istream& in) { read_xml(tag,in);}

  BondOperator(BondOperator const& op, std::string const& t, Parameters const& p)
   : name_(op.name_)
   , term_(t)
   , source_(op.source_)
   , target_(op.target_)
   , parms_(p)
  {}

  void read_xml(const XMLTag&, std::istream&);
  void write_xml(oxstream&) const;

  const std::string& name() const { return name_;}
  const std::string& term () const { return term_;}
  std::string& term () { return term_;}
  const std::string& source () const { return source_;}
  const std::string& target () const { return target_;}
  void substitute_operators(const ModelLibrary& m, const Parameters& p=Parameters());

  template <class T, class I>
  boost::multi_array<std::pair<T,bool>,4> matrix(const SiteBasisDescriptor<I>&, const SiteBasisDescriptor<I>&, const Parameters& =Parameters()) const;

  template <class T, class I>
  std::vector<boost::tuple<expression::Term<T>,SiteOperator,SiteOperator> > templated_split(SiteBasisDescriptor<I> const&, SiteBasisDescriptor<I> const&, const Parameters& = Parameters()) const;

  template <class I>
  std::vector<boost::tuple<Term,SiteOperator,SiteOperator> > split(SiteBasisDescriptor<I> const& b1 ,SiteBasisDescriptor<I> const& b2,const Parameters& p= Parameters()) const
  { return templated_split<std::complex<double> >(b1,b2,p);}

  std::vector<boost::tuple<Term,SiteOperator,SiteOperator> > split(const Parameters& p= Parameters()) const
  { return templated_split<std::complex<double> >(SiteBasisDescriptor<short>(),SiteBasisDescriptor<short>(),p);}
  std::set<std::string> operator_names(const Parameters& = Parameters()) const;

  Parameters const& parms() const { return parms_;}
private:
  std::string name_;
  std::string term_;
  std::string source_;
  std::string target_;
  Parameters parms_;
};



template <class I, class T>
bool BondOperatorSplitter<I,T>::can_evaluate_function(const std::string&name , const expression::Expression<T>& arg,bool isarg) const
{
  return (arg==sites_.first || arg==sites_.second || expression::ParameterEvaluator<T>::can_evaluate_function(name,arg,isarg));
}


template <class I, class T>
expression::Expression<T> BondOperatorSplitter<I,T>::partial_evaluate_function(const std::string& name, const expression::Expression<T>& arg, bool isarg) const
{
  if (arg==sites_.second) {
    site_ops_.second *= expression::Function<T>(name,arg);
    expression::Expression<T> e(second_site_fermionic_ && basis2_.is_fermionic(name) ? -1. : 1.);
    return e;
  }
  else  if (arg==sites_.first) {
    site_ops_.first *= expression::Function<T>(name,arg);
    if (basis1_.is_fermionic(name))
        second_site_fermionic_ = !second_site_fermionic_;
    return expression::Expression<T>(1.);
  }
  return expression::ParameterEvaluator<T>(*this).partial_evaluate_function(name,arg,isarg);
}


template <class I, class T>
boost::multi_array<std::pair<T,bool>,4> get_fermionic_matrix(T,const BondOperator& m, const SiteBasisDescriptor<I>& basis1, const SiteBasisDescriptor<I>& basis2, const Parameters& p=Parameters())
{
  return m.template matrix<T,I>(basis1,basis2,p);
}

template <class T, class I>
boost::multi_array<T,4> get_matrix(T,const BondOperator& m, const SiteBasisDescriptor<I>& basis1, const SiteBasisDescriptor<I>& basis2, const Parameters& p=Parameters())
{
  boost::multi_array<std::pair<T,bool>,4> f_matrix = m.template matrix<T,I>(basis1,basis2,p);
  boost::multi_array<T,4> matrix(boost::extents[f_matrix.shape()[0]][f_matrix.shape()[1]][f_matrix.shape()[2]][f_matrix.shape()[3]]);
  for (int i=0;i<f_matrix.shape()[0];++i)
    for (int j=0;j<f_matrix.shape()[1];++j)
      for (int k=0;k<f_matrix.shape()[2];++k)
        for (int l=0;l<f_matrix.shape()[3];++l)
          if (f_matrix[i][j][k][l].second)
            boost::throw_exception(std::runtime_error("Cannot convert fermionic operator to a bosonic matrix"));
          else
           matrix[i][j][k][l]=f_matrix[i][j][k][l].first;
  return matrix;
}

template <class T, class I> boost::multi_array<std::pair<T,bool>,4>
BondOperator::matrix(const SiteBasisDescriptor<I>& b1,
                              const SiteBasisDescriptor<I>& b2,
                              const Parameters& p) const
{
  typedef typename expression_value_type_traits<T>::value_type value_type;

  SiteBasisDescriptor<I> basis1(b1);
  SiteBasisDescriptor<I> basis2(b2);
  basis1.set_parameters(p);
  basis2.set_parameters(p);
  Parameters parms(p);
  parms.copy_undefined(basis1.get_parameters());
  parms.copy_undefined(basis2.get_parameters());
  std::size_t dim1=basis1.num_states();
  std::size_t dim2=basis2.num_states();
  boost::multi_array<std::pair<T,bool>,4> mat(boost::extents[dim1][dim2][dim1][dim2]);
  // parse expression and store it as sum of terms
  expression::Expression<value_type> ex(term());
  ex.flatten();
  ex.simplify();
  // fill the matrix
    site_basis<I> states1(basis1);
    site_basis<I> states2(basis2);
    for (std::size_t i=0;i<mat.shape()[0];++i)
      for (std::size_t j=0;j<mat.shape()[1];++j)
        for (std::size_t k=0;k<mat.shape()[2];++k)
          for (std::size_t l=0;l<mat.shape()[3];++l)
            mat[i][j][k][l].second=false;
    for (std::size_t i1=0;i1<states1.size();++i1)
      for (std::size_t i2=0;i2<states2.size();++i2) {
      //calculate expression applied to state *it and store it into matrix
        for (typename expression::Expression<value_type>::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
          BondOperatorEvaluator<I,value_type> evaluator(states1[i1], states2[i2], basis1, basis2, source(), target(), parms);
          expression::Term<value_type> term(*tit);
          term.partial_evaluate(evaluator);
          unsigned int j1=states1.index(evaluator.state().first);
          unsigned int j2=states2.index(evaluator.state().second);
          if (is_nonzero(term) && j1<dim1 && j2<dim2) {
            if (is_nonzero(mat[i1][i2][j1][j2].first)) {
              if (mat[i1][i2][j1][j2].second != evaluator.fermionic())
              boost::throw_exception(std::runtime_error("Inconsistent fermionic nature of a matrix element: "
                                    + boost::lexical_cast<std::string>(*tit) + " is inconsistent with "
                                    + boost::lexical_cast<std::string>(mat[i1][i2][j1][j2].first) +
                                    ". Please contact the library authors for an extension to the ALPS model library."));
            }
            else
              mat[i1][i2][j1][j2].second=evaluator.fermionic();
            if (boost::is_arithmetic<T>::value || is_complex<T>::value)
              if (!can_evaluate(boost::lexical_cast<std::string>(term)))
                boost::throw_exception(std::runtime_error("Cannot evaluate expression " + boost::lexical_cast<std::string>(term)));
            mat[i1][i2][j1][j2].first += evaluate<T>(term);
            simplify(mat[i1][i2][j1][j2].first);
        }
      }
    }
  return mat;
}

template <class T, class I>
std::vector<boost::tuple<expression::Term<T>,SiteOperator,SiteOperator> > alps::BondOperator::templated_split(SiteBasisDescriptor<I> const& b1, SiteBasisDescriptor<I> const& b2,const Parameters& p) const
{
  std::vector<boost::tuple<expression::Term<T>,SiteOperator,SiteOperator> > terms;
  Expression ex(term());
  ex.flatten();
  ex.simplify();

  for (typename Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
    BondOperatorSplitter<short> evaluator(b1,b2,source(),target(),p);
    expression::Term<T> term(*tit);
    term.partial_evaluate(evaluator);
    term.simplify();
    terms.push_back(boost::make_tuple(term,
        SiteOperator(boost::lexical_cast<std::string>(evaluator.site_operators().first),source()),
        SiteOperator(boost::lexical_cast<std::string>(evaluator.site_operators().second),target())));
  }
  return terms;
}


} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::BondOperator& q)
{
  q.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::BondOperator& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
