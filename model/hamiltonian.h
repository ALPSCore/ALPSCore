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
#include <alps/model/globaloperator.h>
#include <vector>

namespace alps {

class ModelLibrary;

template<class I>
class HamiltonianDescriptor : public GlobalOperator
{
public:
  typedef std::map<std::string,BasisDescriptor<I> > basis_map;
  typedef std::map<std::string,GlobalOperator> operator_map;
  HamiltonianDescriptor() {}
  HamiltonianDescriptor(const XMLTag&, std::istream&, const basis_map& = basis_map(), const operator_map& = operator_map());
  void write_xml(oxstream&) const;

  const std::string& name() const { return name_;}
  const BasisDescriptor<I>& basis() const { return basis_;}
  BasisDescriptor<I>& basis() { return basis_;}
  
  const Parameters& default_parameters() const { return parms_;}
  bool set_parameters(Parameters p);

  template <class G>
  void create_terms(graph_helper<G> const& l)
  {
    basis_.create_site_bases(l);
    parms_.copy_undefined(GlobalOperator::create_terms(l));
  }

private:
  std::string name_;
  std::string operator_name_;
  std::string basisname_;
  BasisDescriptor<I> basis_;
  Parameters parms_;
};

template <class I>
bool HamiltonianDescriptor<I>::set_parameters(Parameters p)
{
  return basis_.set_parameters(p);
}

#ifndef ALPS_WITHOUT_XML
template <class I>
HamiltonianDescriptor<I>::HamiltonianDescriptor(const XMLTag& intag, std::istream& is, const basis_map& bases, const operator_map& ops)
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
      basis_ = bases.find(basisname_)->second;
      if (tag.type!=XMLTag::SINGLE) {
        tag = parse_tag(is);
        if (tag.name!="/BASIS")
          boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in sitebasis reference"));
      }
    }
    tag = parse_tag(is);
    if (tag.name=="HAMILTONIANOPERATOR") {
      operator_name_=tag.attributes["ref"];
      if (ops.find(operator_name_)==ops.end())
        boost::throw_exception(std::runtime_error("unknown operator: " + operator_name_ + " in <HAMILTONIAN>"));
      static_cast<GlobalOperator&>(*this) = ops.find(operator_name_)->second;
      if (tag.type!=XMLTag::SINGLE) {
        tag = parse_tag(is);
        if (tag.name!="/HAMILTONIANOPERATOR")
          boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in operator reference"));
      }
    }
    else if (tag.name!="/" + intag.name)
      tag = read_xml(tag,is);
    if (tag.name!="/" + intag.name)
      boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in <" + intag.name +">"));
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
  if (operator_name_.empty())
    write_operators_xml(os);
  else
    os << start_tag("HAMILTONIANOPERATOR") << attribute("ref", operator_name_) << end_tag("HAMILTONIANOPERATOR");
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
