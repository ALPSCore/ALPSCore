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
  boost::multi_array<std::string,2> matrix(const SiteBasisDescriptor<I>&, const operator_map& = operator_map());
private:
  int type_;
  std::string term_;
};

template<class I>
class BondTermDescriptor
{
public:
  BondTermDescriptor() : type_(-2) {}
#ifndef ALPS_WITHOUT_XML
  BondTermDescriptor(const XMLTag&, std::istream&);
  void write_xml(std::ostream&, const std::string& = "") const;
#endif

  bool match_type(int type) const { return type_==-1 || type==type_;}
  const std::string& term () const { return term_;}
  const std::string& source () const { return source_;}
  const std::string& target () const { return target_;}
  
 // boost::multi_array<std::string,2> matrix(const SiteBasisDescriptor&, const SiteBasisDescriptor&);
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
private:
  std::string name_;
  std::string basisname_;
  BasisDescriptor<I> basis_;
  std::vector<SiteTermDescriptor<I> > siteterms_;
  std::vector<BondTermDescriptor<I> > bondterms_;
};


template <class I> boost::multi_array<std::string,2> 
SiteTermDescriptor<I>::matrix(const SiteBasisDescriptor<I>& basis, const operator_map& ops)
{
  std::size_t dim=basis.num_elements();
  boost::multi_array<std::string,2> mat(boost::extents[dim][dim]);
  // parse expression and store it as sum of terms
  alps::Expression ex(term());
  
  ex.flatten();
  // fill the matrix
  SiteBasisStates<I> states(basis);
  for (typename SiteBasisStates<I>::const_iterator it=states.begin();it!=states.end();++it) {
    //calculate expression applied to state *it and store it into matrix
    for (alps::Expression::term_iterator iti = ex.terms().first; tit !=ex.terms.second; ++tit) {
      std::pair<StateDescriptor<I>,std::string> res; // = tit->evaluate();
      mat[states.index(res.first)]+=res.second;
    }
  }
  return mat;
}


template <class I>
SiteTermDescriptor<I> site_term(int type) 
{
  for (typename std::vector<SiteTermDescriptor<I> >::const_iterator it =siteterms_.begin();it!=siteterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return SiteTermDescriptor<I>();
}

template <class I>
BondTermDescriptor<I> bond_term(int type) 
{
  for (typename std::vector<BondTermDescriptor<I> >::const_iterator it =bondterms_.begin();it!=bondterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return BondTermDescriptor<I>();
}


#ifndef ALPS_WITHOUT_XML
template <class I>
HamiltonianDescriptor<I>::HamiltonianDescriptor(const XMLTag& intag, std::istream& is, const basis_map& bases) 
{
  XMLTag tag(intag);
  name_=tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
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
