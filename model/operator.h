/***************************************************************************
* ALPS++/model library
*
* model/operator.h    the operator classes
*
* $Id$
*
* Copyright (C) 2003-2003 by Matthias Troyer <troyer@comp-phys.org>,
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
#ifndef ALPS_WITHOUT_XML
  OperatorDescriptor(const XMLTag&, std::istream&);
#endif
  
#ifndef ALPS_WITHOUT_XML
  void write_xml(std::ostream&, const std::string& = "") const;
#endif
  
  std::pair<StateDescriptor<I>,Expression> 
  apply(StateDescriptor<I> state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& p) const;
  
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
#ifndef ALPS_WITHOUT_XML
  SiteTermDescriptor(const XMLTag&, std::istream&);
  void write_xml(std::ostream&, const std::string& = "") const;
#endif

  bool match_type(int type) const { return type_==-1 || type==type_;}
  const std::string& term() const { return term_;}
  boost::multi_array<Expression,2> matrix(const SiteBasisDescriptor<I>&, const operator_map&, 
                                          const Parameters& =Parameters()) const;
private:
  int type_;
  std::string term_;
  Parameters parms_;
};

template<class I>
class BondTermDescriptor
{
public:
  typedef std::map<std::string,OperatorDescriptor<I> > operator_map;
  BondTermDescriptor() : type_(-2) {}
#ifndef ALPS_WITHOUT_XML
  BondTermDescriptor(const XMLTag&, std::istream&);
  void write_xml(std::ostream&, const std::string& = "") const;
#endif

  bool match_type(int type) const { return type_==-1 || type==type_;}
  const std::string& term () const { return term_;}
  const std::string& source () const { return source_;}
  const std::string& target () const { return target_;}
  
  boost::multi_array<Expression,4> matrix(const SiteBasisDescriptor<I>&, const SiteBasisDescriptor<I>&,
                                          const operator_map&, const Parameters& =Parameters()) const;
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
#ifndef ALPS_WITHOUT_XML
  HamiltonianDescriptor(const XMLTag&, std::istream&, const basis_map& = basis_map());
  void write_xml(std::ostream&, const std::string& = "") const;
#endif

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

template <class I>
class SiteOperatorEvaluator : public OperatorEvaluator<I>
{
public:
  typedef typename OperatorEvaluator<I>::operator_map operator_map;

  SiteOperatorEvaluator(const StateDescriptor<I>& s, const SiteBasisDescriptor<I>& b, 
                        const Parameters& p, const operator_map& o) 
    : OperatorEvaluator<I>(p,o), state_(s), basis_(b) {}
  bool can_evaluate(const std::string&) const;
  Expression partial_evaluate(const std::string& name) const;
  const StateDescriptor<I>& state() const { return state_;}
private:
  mutable StateDescriptor<I> state_;
  const SiteBasisDescriptor<I>& basis_;
};

template <class I>
class BondOperatorEvaluator : public OperatorEvaluator<I>
{
public:
  typedef typename OperatorEvaluator<I>::operator_map operator_map;

  BondOperatorEvaluator(StateDescriptor<I> s1, StateDescriptor<I> s2, 
			const SiteBasisDescriptor<I>& b1, const SiteBasisDescriptor<I>& b2, 
			std::string site1, std::string site2,const Parameters& p, const operator_map& o)
    : OperatorEvaluator<I>(p,o), state_(s1,s2), basis1_(b1), basis2_(b2), sites_(site1,site2) {}
  bool can_evaluate_function(const std::string& name, const Expression& argument) const;
  Expression partial_evaluate_function(const std::string& name, const Expression& argument) const;
  const std::pair<StateDescriptor<I>,StateDescriptor<I> >& state() const { return state_;}
private:
  mutable std::pair<StateDescriptor<I>,StateDescriptor<I> > state_;
  const SiteBasisDescriptor<I>& basis1_;
  const SiteBasisDescriptor<I>& basis2_;
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


template <class I>
bool SiteOperatorEvaluator<I>::can_evaluate(const std::string& name) const 
{
  return ops_.find(name) != ops_.end() || ParameterEvaluator::can_evaluate(name);
}

template <class I>
bool BondOperatorEvaluator<I>::can_evaluate_function(const std::string& name, const Expression& arg) const 
{
  return (ops_.find(name) != ops_.end() && (arg== sites_.first || arg==sites_.second)) || 
         ParameterEvaluator::can_evaluate_function(name,arg);
}

template <class I>
Expression SiteOperatorEvaluator<I>::partial_evaluate(const std::string& name) const
{
  typename operator_map::const_iterator op = ops_.find(name);
  if (op!=ops_.end()) {  // evaluate operator
    Expression e;
    boost::tie(state_,e) = op->second.apply(state_,basis_,*this);
    return e;
  }
  else 
    return OperatorEvaluator<I>::partial_evaluate(name);
}

template <class I>
Expression BondOperatorEvaluator<I>::partial_evaluate_function(const std::string& name, const Expression& arg) const
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
std::pair<StateDescriptor<I>,Expression> 
OperatorDescriptor<I>::apply(StateDescriptor<I> state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& eval) const
{
  // set quantum numbers as parameters
  Parameters p=eval.parameters();
  p.copy_undefined(basis.get_parameters());
  for (int i=0;i<basis.size();++i)
    if (p.defined(basis[i].name()))
      boost::throw_exception(std::runtime_error(basis[i].name()+" exists as quantum number and as parameter"));
    else
      p[basis[i].name()]=state[i];
  
  // evaluate matrix element
  Expression e(matrixelement());
  e.partial_evaluate(ParameterEvaluator(p));
  
  // apply operators
  for (int i=0;i<basis.size();++i) {
    const_iterator it=find(basis[i].name());
    if (it!=end()) {
      state[i]+=it->second; // apply change to QN
       if (!basis[i].valid(state[i])) {
         e=Expression(0.);
	 break;
       }
    }
  }
  return std::make_pair(state,e);
}

template <class I> boost::multi_array<Expression,2> 
SiteTermDescriptor<I>::matrix(const SiteBasisDescriptor<I>& basis, const operator_map& ops, const Parameters& p) const
{
  Parameters parms(p);
  parms.copy_undefined(basis.get_parameters());
  std::size_t dim=basis.num_states();
  boost::multi_array<Expression,2> mat(boost::extents[dim][dim]);
  // parse expression and store it as sum of terms
  alps::Expression ex(term());
  ex.flatten();

  // fill the matrix
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
  return mat;
}

template <class I> boost::multi_array<Expression,4> 
BondTermDescriptor<I>::matrix(const SiteBasisDescriptor<I>& basis1, const SiteBasisDescriptor<I>& basis2, 
                              const operator_map& ops, const Parameters& p) const
{
  std::size_t dim1=basis1.num_states();
  std::size_t dim2=basis2.num_states();
  boost::multi_array<Expression,4> mat(boost::extents[dim1][dim2][dim1][dim2]);
  // parse expression and store it as sum of terms
  alps::Expression ex(term());
  ex.flatten();
  // fill the matrix
  SiteBasisStates<I> states1(basis1);
  SiteBasisStates<I> states2(basis2);
  for (int i1=0;i1<states1.size();++i1)
    for (int i2=0;i2<states2.size();++i2)
    {
      //calculate expression applied to state *it and store it into matrix
      for (alps::Expression::term_iterator tit = ex.terms().first; tit !=ex.terms().second; ++tit) {
        BondOperatorEvaluator<I> evaluator(states1[i1],states2[i2],basis1,basis2,
	                                   source(),target(),p,ops);
        Term term(*tit);
        term.partial_evaluate(evaluator);
        int j1=states1.index(evaluator.state().first);
        int j2=states2.index(evaluator.state().second);
        if (boost::lexical_cast<std::string,Term>(term)!="0")
          mat[i1][i2][j1][j2]+=term;
    }
  }
  return mat;
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
void HamiltonianDescriptor<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<HAMILTONIAN";
  if (name()!="")
    os << " name=\"" << name() << "\"";
  os << ">\n";
  for (Parameters::const_iterator it=parms_.begin();it!=parms_.end();++it)
    os << prefix << "  <PARAMETER name=\"" << it->key() << "\" default=\"" << it->value() << "\"/>\n";
  if (basisname_=="")
    basis_.write_xml(os,prefix+"  ");
  else 
    os << prefix << "  <BASIS ref=\"" << basisname_ << "\"/>\n";
  for (typename std::vector<SiteTermDescriptor<I> >::const_iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->write_xml(os,prefix+"  ");
  for (typename std::vector<BondTermDescriptor<I> >::const_iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->write_xml(os,prefix+"  ");
  os << prefix << "</HAMILTONIAN>\n";
}

template <class I>
void OperatorDescriptor<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<OPERATOR name=\"" << name() << "\" matrixelement=\"" << matrixelement() << "\">\n";
  for (const_iterator it=begin();it!=end();++it)
    os << prefix << "  <CHANGE quantumnumber=\"" << it->first << "\" change=\"" << it->second << "\"/>\n";
  os << prefix << "</OPERATOR>\n";
}

template <class I>
void SiteTermDescriptor<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<SITETERM";
  if (type_>=0)
    os << " type=\"" << type_ << "\"";
  if (term()!="")
    os << ">\n" << prefix << "    " << term() << "\n" << prefix << "</SITETERM>\n";
  else
    os << "/>\n";
}

template <class I>
void BondTermDescriptor<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<BONDTERM";
  if (type_>=0)
    os << " type=\"" << type_ << "\"";
  if (term()!="")
    os << " source=\"" << source() << "\" target=\"" << target() << "\">\n"
       << prefix << "    " << term() << "\n" << prefix << "</BONDTERM>\n";
  else
    os << "/>\n";
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

#ifndef ALPS_WITHOUT_XML

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::OperatorDescriptor<I>& q)
{
  q.write_xml(out);
  return out;	
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::HamiltonianDescriptor<I>& q)
{
  q.write_xml(out);
  return out;	
}

#endif

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
