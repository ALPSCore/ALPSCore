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

#ifndef ALPS_MODEL_HAMILTONIAN_H
#define ALPS_MODEL_HAMILTONIAN_H

#include <alps/model/siteterm.h>
#include <alps/model/bondterm.h>
#include <alps/model/basisdescriptor.h>
#include <vector>

namespace alps {

class ModelLibrary;

template<class I>
class HamiltonianDescriptor
{
public:
  typedef std::map<std::string,BasisDescriptor<I> > basis_map;
  HamiltonianDescriptor() {}
  HamiltonianDescriptor(const XMLTag&, std::istream&, const basis_map& = basis_map());
  void write_xml(oxstream&) const;

  const std::string& name() const { return name_;}
  const std::vector<SiteTermDescriptor>& site_terms() const { return siteterms_;}
  const std::vector<BondTermDescriptor>& bond_terms() const { return bondterms_;}
  SiteTermDescriptor site_term(int type=0) const;
  BondTermDescriptor bond_term(int type=0) const;
  const BasisDescriptor<I>& basis() const { return basis_;}
  BasisDescriptor<I>& basis() { return basis_;}
  const Parameters& default_parameters() const { return parms_;}
  bool set_parameters(Parameters p);
  void substitute_operators(const ModelLibrary& m, const Parameters& p);
private:
  std::string name_;
  std::string basisname_;
  BasisDescriptor<I> basis_;
  std::vector<SiteTermDescriptor> siteterms_;
  std::vector<BondTermDescriptor> bondterms_;
  Parameters parms_;
};

template <class I>
void HamiltonianDescriptor<I>::substitute_operators(const ModelLibrary& m, const Parameters& p)
{
  for (std::vector<SiteTermDescriptor>::iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->substitute_operators(m,p);
  for (std::vector<BondTermDescriptor>::iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->substitute_operators(m,p);
}

template <class I>
SiteTermDescriptor HamiltonianDescriptor<I>::site_term(int type) const
{
  for (typename std::vector<SiteTermDescriptor>::const_iterator it =siteterms_.begin();it!=siteterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return SiteTermDescriptor();
}

template <class I>
BondTermDescriptor HamiltonianDescriptor<I>::bond_term(int type) const
{
  for (typename std::vector<BondTermDescriptor>::const_iterator it =bondterms_.begin();it!=bondterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return BondTermDescriptor();
}

template <class I>
bool HamiltonianDescriptor<I>::set_parameters(Parameters p)
{
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
        siteterms_.push_back(SiteTermDescriptor(tag,is));
      else if (tag.name=="BONDTERM")
        bondterms_.push_back(BondTermDescriptor(tag,is));
      else
        boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in <HAMILTONIAN>"));
      tag=parse_tag(is);
    }
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
  for (typename std::vector<SiteTermDescriptor>::const_iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->write_xml(os);
  for (typename std::vector<BondTermDescriptor>::const_iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->write_xml(os);
  os << end_tag("HAMILTONIAN");
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

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
