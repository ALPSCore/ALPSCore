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

#ifndef ALPS_MODEL_BASIS_H
#define ALPS_MODEL_BASIS_H

#include <alps/model/sitebasis.h>
#include <vector>

namespace alps {

template<class I>
class SiteBasisMatch : public SiteBasisDescriptor<I>
{
public:
  typedef SiteBasisDescriptor<I> base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef std::map<std::string,SiteBasisDescriptor<I> > sitebasis_map_type;

  SiteBasisMatch() : type_(-2) {} // matches no site type
#ifndef ALPS_WITHOUT_XML
  SiteBasisMatch(const XMLTag&, std::istream&,const sitebasis_map_type& bases_= sitebasis_map_type());
  void write_xml(std::ostream&, const std::string& = "") const;
#endif
  bool match_type(int type) const { return type_==-1 || type==type_;}
private:
  int type_;
  std::string sitebasis_name_;
};


template<class I>
class BasisDescriptor : public std::vector<SiteBasisMatch<I> >
{
public:
  typedef std::vector<SiteBasisMatch<I> > base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef std::map<std::string,SiteBasisDescriptor<I> > sitebasis_map_type;
  BasisDescriptor() {}
#ifndef ALPS_WITHOUT_XML
  BasisDescriptor(const XMLTag&, std::istream&,const sitebasis_map_type& bases_= sitebasis_map_type());
  void write_xml(std::ostream&, const std::string& = "") const;
#endif
  const SiteBasisDescriptor<I>& site_basis(int type) const;
  const std::string& name() const { return name_;}
private:
  std::string name_;
};

template <class I>
class BasisStatesDescriptor : public std::vector<SiteBasisStates<I> >
{
public:
  typedef std::vector<SiteBasisStates<I> > base_type;
  typedef typename base_type::const_iterator const_iterator;
  BasisStatesDescriptor(const SiteBasisDescriptor<I>& b);
  const SiteBasisDescriptor<I>& basis() const { return basis_;}
private:
  SiteBasisDescriptor<I> basis_;
};



template <class I>
const SiteBasisDescriptor<I>& BasisDescriptor<I>::site_basis(int type) const {
  const_iterator it;
  for (it=begin();it!=end();++it)
    if (it->match_type(type))
      break;
  if (it==end())
    boost::throw_exception("No matching site basis found for site type" << type << "\n");
  return *it;
}


#ifndef ALPS_WITHOUT_XML

template <class I>
SiteBasisMatch<I>::SiteBasisMatch(const XMLTag& intag, std::istream& is, const sitebasis_map_type& bases_) 
{
  XMLTag tag(intag);
  sitebasis_name_ = tag.attributes["ref"];
  type_ = tag.attributes["type"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["type"]);
  if (sitebasis_name_=="") {
    base_type sitebasis(intag,is);
    std::copy(sitebasis.begin(),sitebasis.end(),std::back_inserter(*this));
  }
  else {
    if (bases_.find(sitebasis_name_)==bases_.end())
      boost::throw_exception(std::runtime_error("unknown site basis: " + sitebasis_name_ + " in <BASIS>"));
    else
      static_cast<base_type>(*this) = bases_.find(sitebasis_name_)->second;
    if (tag.type!=XMLTag::SINGLE) {
      tag = parse_tag(is);
      if (tag.name!="/SITEBASIS")
        boost::throw_exception(std::runtime_error("Illegal element name <" + tag.name + "> found in sitebasis reference"));
    }
  }
}

template <class I>
BasisDescriptor<I>::BasisDescriptor(const XMLTag& intag, std::istream& is, const sitebasis_map_type& bases_)
{
  XMLTag tag(intag);
  name_ = tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
    while (tag.name=="SITEBASIS") {
      push_back(SiteBasisMatch<I>(tag,is,bases_));
      tag = parse_tag(is);
    }
    if (tag.name !="/BASIS")
      boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <BASIS> element"));
  }
}

template <class I>
void SiteBasisMatch<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<SITEBASIS";
  if (type_>=0) 
    os << " type=\"" << type_ << "\"";
  if (sitebasis_name_!="")
    os << " ref=\"" << sitebasis_name_ << "\"/>\n";
  else {
    os << ">\n";
    for (const_iterator it=begin();it!=end();++it)
      it->write_xml(os,prefix+"  ");
    os << prefix << "</SITEBASIS>\n";
  }
}

template <class I>
void BasisDescriptor<I>::write_xml(std::ostream& os,  const std::string& prefix) const
{
  os << prefix << "<BASIS name=\"" << name() << "\">\n";
  for (const_iterator it=begin();it!=end();++it)
    it->write_xml(os,prefix+"  ");
  os << prefix << "</BASIS>\n";
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

#ifndef ALPS_WITHOUT_XML

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::SiteBasisMatch<I>& q)
{
  q.write_xml(out);
  return out;	
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::BasisDescriptor<I>& q)
{
  q.write_xml(out);
  return out;	
}

#endif

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
