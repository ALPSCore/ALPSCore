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

#include <alps/model/basisdescriptor.h>
#include <alps/expression.h>
#include <alps/multi_array.hpp>
#include <alps/parameters.h>
#include <boost/tuple/tuple.hpp>
#include <vector>

namespace alps {

template<class I>
class OperatorDescriptor : public std::map<std::string,half_integer<I> >
{
  typedef std::map<std::string,half_integer<I> > super_type;
public:
  typedef typename std::map<std::string,half_integer<I> >::const_iterator const_iterator;
  OperatorDescriptor() {}
  OperatorDescriptor(const std::string& name, const std::string& elm)
    : name_(name), matrixelement_(elm) {}

  OperatorDescriptor(const XMLTag&, std::istream&);

  void write_xml(oxstream&) const;

  template <class STATE>
  boost::tuple<STATE, Expression,bool>
  apply(STATE state, const SiteBasisDescriptor<I>& basis, const ParameterEvaluator& p) const;

  const std::string& name() const { return name_;}
  const std::string& matrixelement() const { return matrixelement_;}

private:
  std::string name_;
  std::string matrixelement_;
};

template <class I>
class OperatorEvaluator : public ParameterEvaluator
{
public:
  typedef ParameterEvaluator super_type;
  typedef std::map<std::string, OperatorDescriptor<I> > operator_map;

  OperatorEvaluator(const Parameters& p, const operator_map& o)
    : ParameterEvaluator(p), ops_(o) {}
  Direction direction() const { return right_to_left; }
  super_type::value_type evaluate(const std::string& name) const
  {
    return partial_evaluate(name).value();
  }
  super_type::value_type evaluate_function(const std::string& name, const Expression& arg) const
  {
    return partial_evaluate_function(name,arg).value();
  }

protected:
  const operator_map& ops_;
};

template <class I>
template <class STATE>
boost::tuple<STATE, Expression,bool>
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
  bool fermion_count=false;
  bool fermionic=false;
  for (int i=0;i<basis.size();++i) {
    const_iterator it=this->find(basis[i].name());
    if (it!=super_type::end()) {
      if (basis[i].fermionic() && is_odd(it->second)) {
        fermionic=!fermionic;
        if (fermion_count)
          e.negate();
      }
      get_quantumnumber(state,i)+=it->second; // apply change to QN
       if (!basis[i].valid(get_quantumnumber(state,i))) {
         e=Expression(0.);
         break;
       }
    }
    if (basis[i].fermionic() && is_odd(get_quantumnumber(state,i)))
      fermion_count=!fermion_count;
  }
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
