/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_SITEOPERATOR_H
#define ALPS_MODEL_SITEOPERATOR_H

#include <alps/model/operator.h>
#include <alps/model/sitestate.h>
#include <alps/model/sitebasisstates.h>
#include <alps/expression.h>
#include <alps/multi_array.hpp>
#include <alps/parameters.h>

namespace alps {

template <class I, class T=std::complex<double>, class STATE = site_state<I> > class SiteOperatorEvaluator;
class ModelLibrary;

class SiteOperator
{
public:
  SiteOperator() {}
  SiteOperator(const std::string& t, const std::string& s="") : term_(t), site_(s) {}
  template <class T>
  SiteOperator(const T& t, const std::string& s="")
    : term_(boost::lexical_cast<std::string>(t)), site_(s) {}
  SiteOperator(const XMLTag& tag, std::istream& is) { read_xml(tag,is);}

  void read_xml(const XMLTag& tag, std::istream& is);
  void write_xml(oxstream&) const;

  const std::string& site() const { return site_;}
  const std::string& term() const { return term_;}
  const std::string& name() const { return name_;}
  template <class I, class T>
  boost::multi_array<std::pair<T,bool>,2> matrix(const SiteBasisDescriptor<I>&,
                                          const Parameters& p=Parameters()) const;

  void substitute_operators(const ModelLibrary& m, const Parameters& p=Parameters());

private:
  std::string term_;
  std::string site_;
  std::string name_;
};

template <class I, class T>
boost::multi_array<T,2> get_matrix(T,const SiteOperator& m, const SiteBasisDescriptor<I>& basis1, const Parameters& p=Parameters())
{
  boost::multi_array<std::pair<T,bool>,2> f_matrix = m.template matrix<I,T>(basis1,p);
  boost::multi_array<T,2> matrix(boost::extents[f_matrix.shape()[0]][f_matrix.shape()[1]]);
  for (int i=0;i<f_matrix.shape()[0];++i)
    for (int j=0;j<f_matrix.shape()[1];++j)
      if (f_matrix[i][j].second)
        boost::throw_exception(std::runtime_error("Cannot convert fermionic operator to a bosonic matrix"));
      else
        matrix[i][j]=f_matrix[i][j].first;
  return matrix;
}

template <class I, class T>
inline boost::multi_array<std::pair<T,bool>,2> get_fermionic_matrix(T,const SiteOperator& m, const SiteBasisDescriptor<I>& basis1,  const Parameters& p=Parameters())
{
  return m.template matrix<I,T>(basis1,p);
}


template <class I, class T> boost::multi_array<std::pair<T,bool>,2>
SiteOperator::matrix(const SiteBasisDescriptor<I>& b,  const Parameters& p) const
{
  typedef typename expression_value_type_traits<T>::value_type value_type;

  SiteBasisDescriptor<I> basis(b);
  basis.set_parameters(p);
  Parameters parms(p);
  parms.copy_undefined(basis.get_parameters());
  std::size_t dim=basis.num_states();
  boost::multi_array<std::pair<T,bool>,2> mat(boost::extents[dim][dim]);
  // parse expression and store it as sum of terms
  expression::Expression<value_type> ex(term());
  ex.flatten();
  ex.simplify();

  // fill the matrix
    site_basis<I> states(basis);
    for (int i=0;i<states.size();++i)
      for (int j=0;j<states.size();++j)
        mat[i][j].second=false;
    for (int i=0;i<states.size();++i) {
    //calculate expression applied to state *it and store it into matrix
      for (typename expression::Expression<value_type>::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
            SiteOperatorEvaluator<I,value_type> evaluator(states[i],basis,parms,site());
        expression::Term<value_type> term(*tit);
        term.partial_evaluate(evaluator);
        unsigned int j = states.index(evaluator.state());
            if (is_nonzero(term)) {
          if (is_nonzero(mat[i][j].first) && j<states.size()) {
            if (mat[i][j].second != evaluator.fermionic())
              boost::throw_exception(std::runtime_error("Inconsistent fermionic nature of a matrix element: "
                                    + boost::lexical_cast<std::string>(*tit) + " is inconsistent with "
                                    + boost::lexical_cast<std::string>(mat[i][j].first) + 
                                    ". Please contact the library authors for an extension to the ALPS model library."));
          }
          else
            mat[i][j].second=evaluator.fermionic();
            if (boost::is_arithmetic<T>::value || TypeTraits<T>::is_complex)
              if (!can_evaluate(boost::lexical_cast<std::string>(term)))
                boost::throw_exception(std::runtime_error("Cannot evaluate expression " + boost::lexical_cast<std::string>(term)));

          mat[i][j].first += evaluate<T>(term);
          simplify(mat[i][j].first);
        }
      }
    }
  return mat;
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& out, const alps::SiteOperator& q)
{
  q.write_xml(out);
  return out;
}

inline std::ostream& operator<<(std::ostream& out, const alps::SiteOperator& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
