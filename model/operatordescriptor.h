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

#ifndef ALPS_MODEL_OPERATORDESCRIPTOR_H
#define ALPS_MODEL_OPERATORDESCRIPTOR_H

#include <alps/expression.h>
#include <alps/model/half_integer.h>
#include <alps/parameters.h>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <map>

namespace alps {

template <class I> class SiteBasisDescriptor;

template <class I>
class OperatorDescriptor : public std::map<std::string,half_integer<I> >
{
  typedef std::map<std::string,half_integer<I> > super_type;
public:
  typedef typename std::map<std::string,half_integer<I> >::const_iterator const_iterator;
  typedef std::map<std::string, OperatorDescriptor<I> > operator_map;

  OperatorDescriptor() {}
  OperatorDescriptor(const std::string& name, const std::string& elm)
    : name_(name), matrixelement_(elm) {}

  OperatorDescriptor(const XMLTag&, std::istream&);

  void write_xml(oxstream&) const;

  template <class STATE>
  boost::tuple<STATE, Expression,bool>
  apply(STATE state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& p) const;
  bool is_fermionic(const SiteBasisDescriptor<I>& basis) const;

  const std::string& name() const { return name_;}
  const std::string& matrixelement() const { return matrixelement_;}

private:
  std::string name_;
  std::string matrixelement_;
};

template <class I>
bool OperatorDescriptor<I>::is_fermionic(const SiteBasisDescriptor<I>& basis) const
{
  // note: we do not check if all QNs changed by the operator are present in the basis since this will
  // anyways be checked for when applyingh the operator later.
  bool fermionic=false;
  for (int i=0;i<basis.size();++i) {
    typename super_type::const_iterator it=super_type::find(basis[i].name());
    if (it!=super_type::end() && basis[i].fermionic() && is_odd(it->second))
      fermionic=!fermionic;
  }
  return fermionic;
}

template <class I>
template <class STATE>
boost::tuple<STATE, Expression,bool>
OperatorDescriptor<I>::apply(STATE state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& eval) const
{
  // set quantum numbers as parameters
  Parameters p=eval.parameters();
  p.copy_undefined(basis.get_parameters());
  for (int i=0;i<basis.size();++i) {
    if (p.defined(basis[i].name()))
      boost::throw_exception(std::runtime_error(basis[i].name()+" exists as quantum number and as parameter"));
    else
      p[basis[i].name()]=get_quantumnumber(state,i);
  }
  // evaluate matrix element
  Expression e(matrixelement());
  e.partial_evaluate(ParameterEvaluator(p));
  // apply operators
  bool fermion_count=false;
  bool fermionic=false;
  int count=0;
  for (int i=0;i<basis.size();++i) {
    const_iterator it=this->find(basis[i].name());
    if (it!=super_type::end()) {
      ++count;
      if (basis[i].fermionic() && is_odd(it->second)) {
        fermionic=!fermionic;
        if (fermion_count)
          e.negate();
      }
      get_quantumnumber(state,i)+=it->second; // apply change to QN
    }
    if (basis[i].fermionic() && is_odd(get_quantumnumber(state,i)))
      fermion_count=!fermion_count;
  }
  if (count != super_type::size())
    boost::throw_exception(std::runtime_error("Not all quantum numbers exist when applying operator " +name()));
  if (!basis.valid(state))
    e=Expression();
  return boost::make_tuple(state,e,fermionic);
}

#ifndef ALPS_WITHOUT_XML

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
void OperatorDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("OPERATOR") << attribute("name", name()) << attribute("matrixelement", matrixelement());
  for (const_iterator it=super_type::begin();it!=super_type::end();++it)
    os << start_tag("CHANGE") << attribute("quantumnumber", it->first)
       << attribute("change", it->second) << end_tag("CHANGE");
  os << end_tag("OPERATOR");
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

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
