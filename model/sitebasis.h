/***************************************************************************
* ALPS++/model library
*
* model/sitebasis.h    the basis classes
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

#ifndef ALPS_MODEL_SITEBASIS_H
#define ALPS_MODEL_SITEBASIS_H

#include <alps/model/quantumnumber.h>
#include <cstddef>
#include <vector>

namespace alps {

template<class I>
class SiteBasisDescriptor : public std::vector<QuantumNumber<I> >
{
public:
  typedef typename std::vector<QuantumNumber<I> >::const_iterator const_iterator;
  
  SiteBasisDescriptor() : num_states_(0) {}
#ifndef ALPS_WITHOUT_XML
  SiteBasisDescriptor(const XMLTag&, std::istream&);
  void write_xml(std::ostream&, const std::string& = "") const;
#endif

  const std::string& name() const { return name_;}
  bool valid(const std::vector<half_integer<I> >&) const;
  std::size_t num_states() const { if (!valid_ && !evaluate()) boost::throw_exception(std::runtime_error("Cannot evaluate quantum numbers in site basis " +name())); return num_states_;}
  bool set_parameters(const Parameters&);
  const Parameters& get_parameters() const { return parms_;}
private:
  mutable bool valid_;
  bool evaluate() const;
  Parameters parms_;
  std::string name_;
  mutable std::size_t num_states_;
};

template <class I>
class StateDescriptor : public std::vector<half_integer<I> > {
public:
  typename std::vector<half_integer<I> >::const_iterator const_iterator;
  StateDescriptor() {}
  StateDescriptor(const std::vector<half_integer<I> >& x) : std::vector<half_integer<I> >(x)  {}
};


template <class I>
class SiteBasisStates : public std::vector<StateDescriptor<I> >
{
public:
  typedef std::vector<StateDescriptor<I> > base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef typename base_type::value_type value_type;
  typedef typename base_type::size_type size_type;
  SiteBasisStates(const SiteBasisDescriptor<I>& b);
  
  size_type index(const value_type& x) const;
  const SiteBasisDescriptor<I>& basis() const { return basis_;}
private:
  SiteBasisDescriptor<I> basis_;
};


// ------------------------------- implementation ----------------------------------

template <class I>
bool SiteBasisDescriptor<I>::valid(const std::vector<half_integer<I> >& x) const
{
  if(!valid_ && !evaluate()) 
    boost::throw_exception(std::runtime_error("Cannot evaluate quantum numbers in site basis " +name()));
  if (size() != x.size())
    return false;
  for (int i=0;i<size();++i)
    if (!(*this)[i].valid(x[i]))
      return false;
  return true;
}


template <class I>
bool SiteBasisDescriptor<I>::set_parameters(const Parameters& p)
{
  for (Parameters::iterator it=parms_.begin();it!=parms_.end();++it)
    if (p.defined(it->key())) 
	it->value() = p[it->key()];
  evaluate();
  return valid_;
}

template <class I>
bool SiteBasisDescriptor<I>::evaluate() const
{
  valid_=true;
  for (const_iterator it=begin();it!=end();++it)
    valid_ = valid_ && const_cast<QuantumNumber<I>&>(*it).set_parameters(parms_);
  if (valid_) {
    num_states_=1;
    for (const_iterator it=begin();it!=end();++it) {
      if(it->levels()==std::numeric_limits<I>::max()) {
        num_states_=std::numeric_limits<I>::max();
        break;
      }
      num_states_ *= it->levels();
    }
  }
  return valid_;
}

template <class I>
typename SiteBasisStates<I>::size_type SiteBasisStates<I>::index(const value_type& x) const
{
  return std::find(begin(),end(),x)-begin();
}


template <class I>
SiteBasisStates<I>::SiteBasisStates(const SiteBasisDescriptor<I>& b)
 : basis_(b)
{
  if (b.num_states()==std::numeric_limits<I>::max())
    boost::throw_exception(std::runtime_error("Cannot build infinite set of basis states\n"));
  std::vector<half_integer<I> > quantumnumbers;
  for (int i=0;i<basis_.size();++i)
    quantumnumbers.push_back(basis_[i].min());
  int i=0;
  if (basis_.valid(quantumnumbers)) 
    do {
      push_back(quantumnumbers);
      i=0;
      while (i<basis_.size()) {
	if(basis_[i].valid(++quantumnumbers[i]))
	  break;
	quantumnumbers[i]=basis_[i].min();
	++i;
      }
    } while (i<basis_.size());
}

#ifndef ALPS_WITHOUT_XML

template <class I>
SiteBasisDescriptor<I>::SiteBasisDescriptor(const XMLTag& intag, std::istream& is)
 : valid_(false)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
    while (tag.name!="/SITEBASIS") {
      if (tag.name=="QUANTUMNUMBER")
        push_back(QuantumNumber<I>(tag,is));
      else if (tag.name=="PARAMETER")
        parms_[tag.attributes["name"]]=tag.attributes["default"];
      if (tag.type!=XMLTag::SINGLE)
        tag = parse_tag(is);
      tag = parse_tag(is);
    }
    if (tag.name !="/SITEBASIS")
      boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <SITEBASIS> element"));
  }
}

template <class I>
void SiteBasisDescriptor<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<SITEBASIS name=\"" << name() <<"\">\n";
  for (Parameters::const_iterator it=parms_.begin();it!=parms_.end();++it)
    os << prefix << "  <PARAMETER name=\"" << it->key() << "\" default=\"" << it->value() << "\"/>\n";
  for (const_iterator it=begin();it!=end();++it)
    it->write_xml(os,prefix+"  ");
  os << prefix << "</SITEBASIS>\n";
}

#endif

}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

#ifndef ALPS_WITHOUT_XML

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::SiteBasisDescriptor<I>& q)
{
  q.write_xml(out);
  return out;	
}

#endif

template <class I>
std::ostream& operator<<(std::ostream& out, const alps::StateDescriptor<I>& s)
{
  out << "|";
  for (typename alps::StateDescriptor<I>::const_iterator it=s.begin();it!=s.end();++it)
    out << *it << " ";
  out << ">\n";
  return out;	
}

template <class I>
std::ostream& operator<<(std::ostream& out, const alps::SiteBasisStates<I>& s)
{
  out << "{\n";
  for (typename alps::SiteBasisStates<I>::const_iterator it=s.begin();it!=s.end();++it) {
    out << "  |";
    for (int i=0;i<s.basis().size();++i)
      out << " " << s.basis()[i].name() << "=" << (*it)[i];
    out << " >\n";
  }
  out << "}\n";
  return out;	
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
