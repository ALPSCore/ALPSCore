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

#ifndef ALPS_MODEL_OPERATOR_H
#define ALPS_MODEL_OPERATOR_H

#include <alps/model/basis.h>
#include <alps/expression.h>
#include <alps/multi_array.hpp>
#include <alps/parameters.h>
#include <boost/tuple/tuple.hpp>
#include <vector>

namespace alps {

template<class I>
class OperatorDescriptor : public std::map<std::string,half_integer<I> >
{
public:
  typedef typename std::map<std::string,half_integer<I> >::const_iterator const_iterator;
  OperatorDescriptor() {}
  OperatorDescriptor(const std::string& name, const std::string& elm)
    : name_(name), matrixelement_(elm) {}

  OperatorDescriptor(const XMLTag&, std::istream&);

  void write_xml(oxstream&) const;

  template <class STATE>
  std::pair<STATE,Expression>
  apply(STATE state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& p) const;

  const std::string& name() const { return name_;}
  const std::string& matrixelement() const { return matrixelement_;}

private:
  std::string name_;
  std::string matrixelement_;
};

template<class I>
class SiteTermDescriptor
{
public:
  typedef std::map<std::string,OperatorDescriptor<I> > operator_map;

  SiteTermDescriptor() : type_(-2) {}
  SiteTermDescriptor(const std::string& t) : type_(-2), term_(t) {}
  SiteTermDescriptor(const Term& t) : type_(-2),
    term_(boost::lexical_cast<std::string>(t)) {}
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
      source_("i"), target("j") {}
  BondTermDescriptor(const Term& term, const std::string& source,
                     const std::string& target)
    : type_(-2), term_(boost::lexical_cast<std::string>(term)),
      source_(source), target(target) {}
  BondTermDescriptor(const XMLTag&, std::istream&);
  void write_xml(oxstream&) const;

  bool match_type(int type) const { return type_==-1 || type==type_;}
  const std::string& term () const { return term_;}
  const std::string& source () const { return source_;}
  const std::string& target () const { return target_;}

  template <class T>
  boost::multi_array<T,4> matrix(const SiteBasisDescriptor<I>&, const SiteBasisDescriptor<I>&,
                                          const operator_map&, const Parameters& =Parameters()) const;
  std::set<Term> split(const operator_map&, const Parameters& =Parameters()) const;

private:
  int type_;
  std::string term_;
  std::string source_;
  std::string target_;
};

template<class I>
class HamiltonianDescriptor
{
public:
  typedef std::map<std::string,BasisDescriptor<I> > basis_map;
  HamiltonianDescriptor() {}
  HamiltonianDescriptor(const XMLTag&, std::istream&, const basis_map& = basis_map());
  void write_xml(oxstream&) const;

  const std::string& name() const { return name_;}
  const std::vector<SiteTermDescriptor<I> >& site_terms() const { return siteterms_;}
  const std::vector<BondTermDescriptor<I> >& bond_terms() const { return bondterms_;}
  SiteTermDescriptor<I> site_term(int type=0) const;
  BondTermDescriptor<I> bond_term(int type=0) const;
  const BasisDescriptor<I>& basis() const { return basis_;}
  BasisDescriptor<I>& basis() { return basis_;}
  const Parameters& default_parameters() const { return parms_;}
  bool set_parameters(Parameters p);
private:
  std::string name_;
  std::string basisname_;
  BasisDescriptor<I> basis_;
  std::vector<SiteTermDescriptor<I> > siteterms_;
  std::vector<BondTermDescriptor<I> > bondterms_;
  Parameters parms_;
};

template <class I>
class OperatorEvaluator : public ParameterEvaluator
{
public:
  typedef std::map<std::string,OperatorDescriptor<I> > operator_map;

  OperatorEvaluator(const Parameters& p, const operator_map& o)
    : ParameterEvaluator(p), ops_(o) {}
  Direction direction() const { return right_to_left;}
  double evaluate(const std::string& name) const;
  double evaluate_function(const std::string& name, const Expression& arg) const;

protected:
  const operator_map& ops_;
};

template <class I, class STATE=StateDescriptor<I> >
class SiteOperatorEvaluator : public OperatorEvaluator<I>
{
public:
  typedef typename OperatorEvaluator<I>::operator_map operator_map;
  typedef STATE state_type;

  SiteOperatorEvaluator(const state_type& s, const SiteBasisDescriptor<I>& b,
                        const Parameters& p, const operator_map& o)
    : OperatorEvaluator<I>(p,o), state_(s), basis_(b) {}
  bool can_evaluate(const std::string&) const;
  Expression partial_evaluate(const std::string& name) const;
  const state_type& state() const { return state_;}
private:
  mutable state_type state_;
  const SiteBasisDescriptor<I>& basis_;
};

template <class I, class STATE1=StateDescriptor<I>, class STATE2=StateDescriptor<I>  >
class BondOperatorEvaluator : public OperatorEvaluator<I>
{
public:
  typedef typename OperatorEvaluator<I>::operator_map operator_map;

  BondOperatorEvaluator(const STATE1& s1, const STATE2& s2,
                        const SiteBasisDescriptor<I>& b1, const SiteBasisDescriptor<I>& b2,
                        std::string site1, std::string site2,const Parameters& p, const operator_map& o)
    : OperatorEvaluator<I>(p,o), state_(s1,s2), basis1_(b1), basis2_(b2), sites_(site1,site2) {}
  bool can_evaluate_function(const std::string& name, const Expression& argument) const;
  Expression partial_evaluate_function(const std::string& name, const Expression& argument) const;
  const std::pair<STATE1,STATE2>& state() const { return state_;}
private:
  mutable std::pair<STATE1,STATE2> state_;
  const SiteBasisDescriptor<I>& basis1_;
  const SiteBasisDescriptor<I>& basis2_;
  std::pair<std::string,std::string> sites_;
};

template <class I>
class BondOperatorSplitter : public OperatorEvaluator<I>
{
 public:
  typedef typename OperatorEvaluator<I>::operator_map operator_map;

  BondOperatorSplitter(std::string site1, std::string site2,
                       const Parameters& p, const operator_map& o)
    : OperatorEvaluator<I>(p,o), sites_(site1,site2) {}

  bool can_evaluate_function(const std::string& name, const Expression& argument) const;
  Expression partial_evaluate_function(const std::string& name, const Expression& argument) const;
  const std::pair<Term,Term>& site_operators() const { return site_ops_;}
 private:
  mutable std::pair<Term,Term> site_ops_;
  std::pair<std::string,std::string> sites_;
};


template <class I>
double OperatorEvaluator<I>::evaluate(const std::string& name) const
{
  return partial_evaluate(name).value();
}

template <class I>
double OperatorEvaluator<I>::evaluate_function(const std::string& name, const Expression& arg) const
{
  return partial_evaluate_function(name,arg).value();
}


template <class I, class STATE>
bool SiteOperatorEvaluator<I,STATE>::can_evaluate(const std::string& name) const
{
  if (ops_.find(name) != ops_.end()) {
    SiteOperatorEvaluator<I,STATE> eval(*this);
    return eval.partial_evaluate(name).can_evaluate(ParameterEvaluator(*this));
  }
  else
    return ParameterEvaluator::can_evaluate(name);
}

template <class I, class STATE1, class STATE2>
bool BondOperatorEvaluator<I,STATE1,STATE2>::can_evaluate_function(const std::string& name, const Expression& arg) const
{
  if (ops_.find(name) != ops_.end() && (arg== sites_.first || arg==sites_.second)) {
    BondOperatorEvaluator<I,STATE1,STATE2> eval(*this);
    return eval.partial_evaluate_function(name,arg).can_evaluate(ParameterEvaluator(*this));
  }
  else
    return ParameterEvaluator::can_evaluate_function(name,arg);
}

template <class I>
bool BondOperatorSplitter<I>::can_evaluate_function(const std::string& name, const Expression& arg) const
{
  return (ops_.find(name) != ops_.end() && (arg== sites_.first || arg==sites_.second)) ||
         ParameterEvaluator::can_evaluate_function(name,arg);
}

template <class I,class STATE>
Expression SiteOperatorEvaluator<I,STATE>::partial_evaluate(const std::string& name) const
{
  typename operator_map::const_iterator op = ops_.find(name);
  if (op!=ops_.end()) {  // evaluate operator
    Expression e;
    boost::tie(state_,e) = op->second.apply(state_,basis_,ParameterEvaluator(*this));
    return e;
  }
  else
    return OperatorEvaluator<I>::partial_evaluate(name);
}

template <class I, class STATE1, class STATE2>
Expression BondOperatorEvaluator<I,STATE1,STATE2>::partial_evaluate_function(const std::string& name, const Expression& arg) const
{
  typename operator_map::const_iterator op = ops_.find(name);
  if (op!=ops_.end()) {  // evaluate operator
    Expression e;
    if (arg==sites_.first)
      boost::tie(state_.first,e) = op->second.apply(state_.first,basis1_,*this);
    else  if (arg==sites_.second)
      boost::tie(state_.second,e) = op->second.apply(state_.second,basis2_,*this);
    else
      return ParameterEvaluator(*this).partial_evaluate_function(name,arg);
    return e;
  }
  else
    return ParameterEvaluator(*this).partial_evaluate_function(name,arg);
}

template <class I>
Expression BondOperatorSplitter<I>::partial_evaluate_function(const std::string& name, const Expression& arg) const
{
  typename operator_map::const_iterator op = ops_.find(name);
  if (op!=ops_.end()) {  // evaluate operator
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


template <class I> template <class STATE>
std::pair<STATE,Expression>
OperatorDescriptor<I>::apply(STATE state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& eval) const
{
  // set quantum numbers as parameters
  Parameters p=eval.parameters();
  p.copy_undefined(basis.get_parameters());
  for (int i=0;i<basis.size();++i)
    if (p.defined(basis[i].name()))
      boost::throw_exception(std::runtime_error(basis[i].name()+" exists as quantum number and as parameter"));
    else
      p[basis[i].name()]=get_quantumnumber(state,i);

  // evaluate matrix element
  Expression e(matrixelement());
  e.partial_evaluate(ParameterEvaluator(p));

  // apply operators
  for (int i=0;i<basis.size();++i) {
    const_iterator it=this->find(basis[i].name());
    if (it!=end()) {
      get_quantumnumber(state,i)+=it->second; // apply change to QN
       if (!basis[i].valid(get_quantumnumber(state,i))) {
         e=Expression(0.);
         break;
       }
    }
  }
  return std::make_pair(state,e);
}

template <class I, class T>
boost::multi_array<T,2> get_matrix(T,const SiteTermDescriptor<I>& m, const SiteBasisDescriptor<I>& basis1, const typename SiteTermDescriptor<I>::operator_map& ops, const Parameters& p=Parameters())
{
  return m.template matrix<T>(basis1,ops,p);
}


template <class I> template <class T> boost::multi_array<T,2>
SiteTermDescriptor<I>::matrix(const SiteBasisDescriptor<I>& b, const operator_map& ops, const Parameters& p) const
{
  SiteBasisDescriptor<I> basis(b);
  basis.set_parameters(p);
  Parameters parms(p);
  parms.copy_undefined(basis.get_parameters());
  std::size_t dim=basis.num_states();
  boost::multi_array<T,2> mat(boost::extents[dim][dim]);
  // parse expression and store it as sum of terms
  alps::Expression ex(term());
  ex.flatten();

  // fill the matrix
  if (basis.size()==1) {
    typedef SingleQNStateDescriptor<I> state_type;
    SiteBasisStates<I,state_type> states(basis);
    for (int i=0;i<states.size();++i) {
      //calculate expression applied to state *it and store it into matrix
      for (alps::Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
        SiteOperatorEvaluator<I,state_type> evaluator(states[i],basis,parms,ops);
        Term term(*tit);
        term.partial_evaluate(evaluator);
        int j=states.index(evaluator.state());
        if (boost::lexical_cast<std::string,Term>(term)!="0")
          mat[i][j]+=term;
      }
    }
  }
  else {
    SiteBasisStates<I> states(basis);
    for (int i=0;i<states.size();++i) {
    //calculate expression applied to state *it and store it into matrix
      for (alps::Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
        SiteOperatorEvaluator<I> evaluator(states[i],basis,parms,ops);
        Term term(*tit);
        term.partial_evaluate(evaluator);
        int j=states.index(evaluator.state());
        if (boost::lexical_cast<std::string,Term>(term)!="0")
          mat[i][j]+=term;
      }
    }
  }
  return mat;
}

template <class I, class T>
boost::multi_array<T,4> get_matrix(T,const BondTermDescriptor<I>& m, const SiteBasisDescriptor<I>& basis1, const SiteBasisDescriptor<I>& basis2, const  typename BondTermDescriptor<I>::operator_map& ops, const Parameters& p=Parameters())
{
  return m.template matrix<T>(basis1,basis2,ops,p);
}

template <class I> template <class T> boost::multi_array<T,4>
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
  boost::multi_array<T,4> mat(boost::extents[dim1][dim2][dim1][dim2]);
  // parse expression and store it as sum of terms
  alps::Expression ex(term());
  ex.flatten();
  // fill the matrix
  if (basis1.size()==1 && basis2.size()==1) {
    typedef SingleQNStateDescriptor<I> state_type;
    SiteBasisStates<I,state_type> states1(basis1);
    SiteBasisStates<I,state_type> states2(basis2);
    for (int i1=0;i1<states1.size();++i1)
      for (int i2=0;i2<states2.size();++i2) {
      //calculate expression applied to state *it and store it into matrix
        for (alps::Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
          BondOperatorEvaluator<I,state_type,state_type> evaluator(states1[i1],states2[i2],basis1,basis2,
                                           source(),target(),parms,ops);
          Term term(*tit);
          term.partial_evaluate(evaluator);
          int j1=states1.index(evaluator.state().first);
          int j2=states2.index(evaluator.state().second);
          if (boost::lexical_cast<std::string,Term>(term)!="0")
            mat[i1][i2][j1][j2]+=term;
      }
    }
  }
  else  {
    SiteBasisStates<I> states1(basis1);
    SiteBasisStates<I> states2(basis2);
    for (int i1=0;i1<states1.size();++i1)
      for (int i2=0;i2<states2.size();++i2) {
      //calculate expression applied to state *it and store it into matrix
        for (alps::Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
          BondOperatorEvaluator<I> evaluator(states1[i1],states2[i2],basis1,basis2,
                                           source(),target(),parms,ops);
          Term term(*tit);
          term.partial_evaluate(evaluator);
          int j1=states1.index(evaluator.state().first);
          int j2=states2.index(evaluator.state().second);
          if (boost::lexical_cast<std::string,Term>(term)!="0")
            mat[i1][i2][j1][j2]+=term;
      }
    }
  }
  return mat;
}

template <class I>
std::set<Term> BondTermDescriptor<I>::split(const operator_map& ops, const Parameters& p) const
{
  std::set<Term> terms;
  alps::Expression ex(term());
  ex.flatten();
  for (alps::Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
    BondOperatorSplitter<I> evaluator(source(),target(),p,ops);
    Term term(*tit);
    term.partial_evaluate(evaluator);
    terms.insert(evaluator.site_operators().first);
    terms.insert(evaluator.site_operators().second);
  }
  return terms;
}

template <class I>
SiteTermDescriptor<I> HamiltonianDescriptor<I>::site_term(int type) const
{
  for (typename std::vector<SiteTermDescriptor<I> >::const_iterator it =siteterms_.begin();it!=siteterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return SiteTermDescriptor<I>();
}

template <class I>
BondTermDescriptor<I>  HamiltonianDescriptor<I>::bond_term(int type) const
{
  for (typename std::vector<BondTermDescriptor<I> >::const_iterator it =bondterms_.begin();it!=bondterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return BondTermDescriptor<I>();
}

template <class I>
bool HamiltonianDescriptor<I>::set_parameters(Parameters p)
{
  p.copy_undefined(parms_);
  return basis_.set_parameters(p);
}

#ifndef ALPS_WITHOUT_XML
template <class I>
HamiltonianDescriptor<I>::HamiltonianDescriptor(const XMLTag& intag, std::istream& is, const basis_map& bases)
{
  XMLTag tag(intag);
  name_=tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
    while (tag.name=="PARAMETER") {
      parms_[tag.attributes["name"]]=tag.attributes["default"];
      if (tag.type!=XMLTag::SINGLE) {
        tag=parse_tag(is);
        if (tag.name!="/PARAMETER")
          boost::throw_exception(std::runtime_error("End tag </PARAMETER> missing while parsing " + name() + " Hamiltonian"));
      }
      tag=parse_tag(is);
    }
    if (tag.name!="BASIS")
      boost::throw_exception(std::runtime_error("unexpected element: " + tag.name + " in <HAMILTONIAN>"));
    basisname_ = tag.attributes["ref"];
    if (basisname_=="")
      basis_ =BasisDescriptor<I>(intag,is);
    else {
      if (bases.find(basisname_)==bases.end())
        boost::throw_exception(std::runtime_error("unknown basis: " + basisname_ + " in <HAMILTONIAN>"));
      else
        basis_ = bases.find(basisname_)->second;
      if (tag.type!=XMLTag::SINGLE) {
        tag = parse_tag(is);
        if (tag.name!="/BASIS")
          boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in sitebasis reference"));
      }
    }
    tag = parse_tag(is);
    while (tag.name!="/HAMILTONIAN") {
      if (tag.name=="SITETERM")
        siteterms_.push_back(SiteTermDescriptor<I>(tag,is));
      else if (tag.name=="BONDTERM")
        bondterms_.push_back(BondTermDescriptor<I>(tag,is));
      else
        boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in <HAMILTONIAN>"));
      tag=parse_tag(is);
    }
  }
}

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
OperatorDescriptor<I>::OperatorDescriptor(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  matrixelement_ = tag.attributes["matrixelement"];
  if (name_=="" || matrixelement_=="")
    boost::throw_exception(std::runtime_error("name and matrix element need to be given for <OPERATOR>"));
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
    while (tag.name=="CHANGE") {
      (*this)[tag.attributes["quantumnumber"]]=
        boost::lexical_cast<half_integer<I>,std::string>(tag.attributes["change"]);
      if (tag.type!=XMLTag::SINGLE) {
        tag = parse_tag(is);
        if (tag.name !="/CHANGE")
          boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <OPERATOR> element."));
        }
      tag = parse_tag(is);
    }
    if (tag.name !="/OPERATOR")
      boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <OPERATOR> element"));
  }
}

template <class I>
void HamiltonianDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("HAMILTONIAN");
  if (name()!="")
    os << attribute("name", name());
  for (Parameters::const_iterator it=parms_.begin();it!=parms_.end();++it)
    os << start_tag("PARAMETER") << attribute("name", it->key())
       << attribute("default", it->value()) << end_tag("PARAMETER");
  if (basisname_=="")
    os << basis_;
  else
    os << start_tag("BASIS") << attribute("ref", basisname_) << end_tag("BASIS");
  for (typename std::vector<SiteTermDescriptor<I> >::const_iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->write_xml(os);
  for (typename std::vector<BondTermDescriptor<I> >::const_iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->write_xml(os);
  os << end_tag("HAMILTONIAN");
}

template <class I>
void OperatorDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("OPERATOR") << attribute("name", name()) << attribute("matrixelement", matrixelement());
  for (const_iterator it=begin();it!=end();++it)
    os << start_tag("CHANGE") << attribute("quantumnumber", it->first)
       << attribute("change", it->second) << end_tag("CHANGE");
  os << end_tag("OPERATOR");
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
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::OperatorDescriptor<I>& q)
{
  q.write_xml(out);
  return out;
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::OperatorDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

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

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::HamiltonianDescriptor<I>& q)
{
  q.write_xml(out);
  return out;
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::HamiltonianDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
