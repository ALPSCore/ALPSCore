/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_MODEL_OPERATORDESCRIPTOR_H
#define ALPS_MODEL_OPERATORDESCRIPTOR_H

#include <alps/expression.h>
#include <alps/model/half_integer.h>
#include <alps/parameter.h>
#include <boost/tuple/tuple.hpp>
#include <vector>
#include <map>

namespace alps {

template <class I> class SiteBasisDescriptor;

template <class I>
class OperatorDescriptor : private std::vector<std::pair<std::string,half_integer<I> > >
{
  typedef std::vector<std::pair<std::string,half_integer<I> > > super_type;
public:
  typedef typename super_type::const_iterator const_iterator;
  typedef std::map<std::string, OperatorDescriptor<I> > operator_map;

  OperatorDescriptor() {}
  OperatorDescriptor(const std::string& name, const std::string& elm)
    : name_(name), matrixelement_(elm) {}

  OperatorDescriptor(const XMLTag&, std::istream&);

  void write_xml(oxstream&) const;

  template <class STATE, class T>
  boost::tuple<STATE, expression::Expression<T>,bool>
  apply(STATE state, const SiteBasisDescriptor<I>& basis, const expression::ParameterEvaluator<T>& p, bool) const;
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
  for (int i=0;i<(int)basis.size();++i) {
    for (const_iterator it=this->begin(); it !=this->end();++it) {
      if (it->first == basis[i].name()) {
        if (basis[i].fermionic() && is_odd(it->second))
          fermionic=!fermionic;
        break;
      }
    }
  }
  return fermionic;
}

template <class I>
template <class STATE, class T>
boost::tuple<STATE, expression::Expression<T>,bool>
OperatorDescriptor<I>::apply(STATE state, const SiteBasisDescriptor<I>& basis, const expression::ParameterEvaluator<T>& eval, bool isarg) const
{
  // set quantum numbers as parameters
  Parameters p=eval.parameters();
  p.copy_undefined(basis.get_parameters(false));
  for (std::size_t i=0;i<basis.size();++i) {
    if (p.defined(basis[i].name()))
      boost::throw_exception(std::runtime_error(basis[i].name()+" exists as quantum number and as parameter"));
    else
      p[basis[i].name()]=get_quantumnumber(state,i);
  }
  // evaluate matrix element
  expression::Expression<T> e(matrixelement());
  e.partial_evaluate(expression::ParameterEvaluator<T>(p));
  // apply operators
  bool fermionic=false;
  for (const_iterator it=this->begin(); it !=this->end();++it) {
    bool fermion_count=false;
    std::size_t i;
    for (i=0;i<basis.size();++i) {
      if (it->first == basis[i].name()) {
        if (basis[i].fermionic() && is_odd(it->second)) {
          fermionic=!fermionic;
          if (fermion_count)
            e.negate();
        }
        if (isarg && (it->second!=0))
          boost::throw_exception(std::runtime_error("Cannot apply offdiagonal operator inside function argument or power"));
        get_quantumnumber(state,i)+=it->second; // apply change to QN
        break;
      }
      else if (basis[i].fermionic() && is_odd(get_quantumnumber(state,i)))
        fermion_count=!fermion_count;
    }
    if (i>=basis.size())
      boost::throw_exception(std::runtime_error("Not all quantum numbers exist when applying operator " +name()));
  }
  if (!basis.valid(state))
    return boost::make_tuple(state,expression::Expression<T>(),false);
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
      this->push_back(std::make_pair(tag.attributes["quantumnumber"],
        boost::lexical_cast<half_integer<I>,std::string>(tag.attributes["change"])));
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
  for (const_iterator it=this->begin();it!=this->end();++it)
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
