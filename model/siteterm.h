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

#ifndef ALPS_MODEL_SITETERM_H
#define ALPS_MODEL_SITETERM_H

#include <alps/model/operator.h>
#include <alps/expression.h>
#include <alps/multi_array.hpp>
#include <alps/parameters.h>

namespace alps {

template<class I>
class SiteTermDescriptor
{
public:
  typedef std::map<std::string,OperatorDescriptor<I> > operator_map;

  SiteTermDescriptor() : type_(-2) {}
  SiteTermDescriptor(const std::string& t) : type_(-2), term_(t) {}
#ifndef ALPS_WITH_NEW_EXPRESSION
  SiteTermDescriptor(const Term& t) : type_(-2),
    term_(boost::lexical_cast<std::string>(t)) {}
#else
  template<class T>
  SiteTermDescriptor(const Term<T>& t) : type_(-2),
    term_(boost::lexical_cast<std::string>(t)) {}
#endif // ! ALPS_WITH_NEW_EXPRESSION
  SiteTermDescriptor(const XMLTag&, std::istream&);

  void write_xml(oxstream&) const;

  bool match_type(int type) const { return type_==-1 || type==type_;}
  const std::string& term() const { return term_;}
  template <class T>
  boost::multi_array<T,2> matrix(const SiteBasisDescriptor<I>&, const operator_map&,
                                          const Parameters& =Parameters()) const;
private:
  int type_;
  std::string term_;
};


#ifndef ALPS_WITH_NEW_EXPRESSION
template <class I, class STATE = site_state<I> >
class SiteOperatorEvaluator : public OperatorEvaluator<I>
#else
template <class I, class T, class STATE = site_state<I> >
class SiteOperatorEvaluator : public OperatorEvaluator<I, T>
#endif
{
private:
#ifndef ALPS_WITH_NEW_EXPRESSION
  typedef OperatorEvaluator<I> super_type;
  typedef SiteOperatorEvaluator<I, STATE> SELF_;
  typedef OperatorEvaluator<I> BASE_;
  typedef Expression Expression_;
  typedef ParameterEvaluator ParameterEvaluator_;
#else
  typedef OperatorEvaluator<I, T> super_type;
  typedef SiteOperatorEvaluator<I, T, STATE> SELF_;
  typedef OperatorEvaluator<I, T> BASE_;
  typedef Expression<T> Expression_;
  typedef ParameterEvaluator<T> ParameterEvaluator_;
#endif

public:
  typedef typename BASE_::operator_map operator_map;
  typedef STATE state_type;

  SiteOperatorEvaluator(const state_type& s, const SiteBasisDescriptor<I>& b,
                        const Parameters& p, const operator_map& o)
    : BASE_(p,o), state_(s), basis_(b) {}
  bool can_evaluate(const std::string&) const;
  Expression_ partial_evaluate(const std::string& name) const;
  const state_type& state() const { return state_;}

private:
  mutable state_type state_;
  const SiteBasisDescriptor<I>& basis_;
};


#ifndef ALPS_WITH_NEW_EXPRESSION
template <class I, class STATE>
bool SiteOperatorEvaluator<I, STATE>::can_evaluate(const std::string& name) const
#else
template <class I, class T, class STATE>
bool SiteOperatorEvaluator<I, T, STATE>::can_evaluate(const std::string& name) const
#endif
{
  if (super_type::ops_.find(name) != super_type::ops_.end()) {
    SELF_ eval(*this);
    return eval.partial_evaluate(name).can_evaluate(ParameterEvaluator_(*this));
  }
  else
    return ParameterEvaluator_::can_evaluate(name);
}

#ifndef ALPS_WITH_NEW_EXPRESSION
template <class I, class STATE>
Expression SiteOperatorEvaluator<I, STATE>::partial_evaluate(const std::string& name) const
#else
template <class I, class T, class STATE>
Expression<T> SiteOperatorEvaluator<I, T, STATE>::partial_evaluate(const std::string& name) const
#endif
{
  typename operator_map::const_iterator op = super_type::ops_.find(name);
  if (op!=super_type::ops_.end()) {  // evaluate operator
    Expression_ e;
    boost::tie(state_,e) = op->second.apply(state_, basis_, ParameterEvaluator_(*this));
    return e;
  }
  else
    return BASE_::partial_evaluate(name);
}


template <class I, class T>
boost::multi_array<T,2> get_matrix(T,const SiteTermDescriptor<I>& m, const SiteBasisDescriptor<I>& basis1, const typename SiteTermDescriptor<I>::operator_map& ops, const Parameters& p=Parameters())
{
  return m.template matrix<T>(basis1,ops,p);
}


template <class I> template <class T> boost::multi_array<T,2>
SiteTermDescriptor<I>::matrix(const SiteBasisDescriptor<I>& b,
			      const operator_map& ops,
			      const Parameters& p) const
{
#ifndef ALPS_WITH_NEW_EXPRESSION
  typedef alps::Expression Expression_;
  typedef alps::Term Term_;
#else
  typedef typename alps::expression<T>::type Expression_;
  typedef typename alps::expression<T>::term_type Term_;
#endif

  SiteBasisDescriptor<I> basis(b);
  basis.set_parameters(p);
  Parameters parms(p);
  parms.copy_undefined(basis.get_parameters());
  std::size_t dim=basis.num_states();
  boost::multi_array<T,2> mat(boost::extents[dim][dim]);
  // parse expression and store it as sum of terms
  Expression_ ex(term());
  ex.flatten();

  // fill the matrix
  if (basis.size()==1) {
    typedef single_qn_site_state<I> state_type;
    site_basis<I,state_type> states(basis);
    for (int i=0;i<states.size();++i) {
      //calculate expression applied to state *it and store it into matrix
      for (typename Expression_::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
#ifndef ALPS_WITH_NEW_EXPRESSION
	SiteOperatorEvaluator<I, state_type> evaluator(states[i], basis,parms,ops);
#else
	SiteOperatorEvaluator<I, typename Expression_::value_type, state_type> evaluator(states[i], basis,parms,ops);
#endif
        Term_ term(*tit);
        term.partial_evaluate(evaluator);
        int j = states.index(evaluator.state());
#ifndef ALPS_WITH_NEW_EXPRESSION
	if (boost::lexical_cast<std::string,Term_>(term)!="0")
          mat[i][j] += term;
#else
	if (alps::is_nonzero(term))
	  mat[i][j] += evaluate<T>(term);
#endif
      }
    }
  }
  else {
    site_basis<I> states(basis);
    for (int i=0;i<states.size();++i) {
    //calculate expression applied to state *it and store it into matrix
      for (typename Expression_::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
#ifndef ALPS_WITH_NEW_EXPRESSION
	SiteOperatorEvaluator<I> evaluator(states[i], basis,parms,ops);
#else
	SiteOperatorEvaluator<I, typename Expression_::value_type> evaluator(states[i], basis,parms,ops);
#endif
        Term_ term(*tit);
        term.partial_evaluate(evaluator);
        int j = states.index(evaluator.state());
#ifndef ALPS_WITH_NEW_EXPRESSION
	if (boost::lexical_cast<std::string,Term_>(term)!="0")
          mat[i][j] += term;
#else
	if (alps::is_nonzero(term))
	  mat[i][j] += evaluate<T>(term);
#endif
      }
    }
  }
  return mat;
}

#ifndef ALPS_WITHOUT_XML

template <class I>
SiteTermDescriptor<I>::SiteTermDescriptor(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  type_ = tag.attributes["type"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["type"]);
  if (tag.type!=XMLTag::SINGLE) {
    term_=parse_content(is);
    tag = parse_tag(is);
    if (tag.name !="/SITETERM")
      boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <SITETERM> element"));
  }
}

template <class I>
void SiteTermDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("SITETERM");
  if (type_>=0)
    os << attribute("type", type_);
  if (term()!="")
    os << term();
  os << end_tag("SITETERM");
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::SiteTermDescriptor<I>& q)
{
  q.write_xml(out);
  return out;
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::SiteTermDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
