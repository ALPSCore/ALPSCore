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
#include <alps/lattice/lattice.h>
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
  typedef typename base_type::iterator iterator;
  typedef typename base_type::const_iterator const_iterator;
  typedef std::map<std::string,SiteBasisDescriptor<I> > sitebasis_map_type;
  BasisDescriptor() {}
#ifndef ALPS_WITHOUT_XML
  BasisDescriptor(const XMLTag&, std::istream&,const sitebasis_map_type& bases_= sitebasis_map_type());
  void write_xml(std::ostream&, const std::string& = "") const;
#endif
  const SiteBasisDescriptor<I>& site_basis(int type=0) const;
  const std::string& name() const { return name_;}
  bool set_parameters(const Parameters& p);
private:
  std::string name_;
};

template <class I>
class BasisStatesDescriptor : public std::vector<SiteBasisStates<I> >
{
public:
  typedef std::vector<SiteBasisStates<I> > base_type;
  typedef typename base_type::const_iterator const_iterator;
  template <class G> BasisStatesDescriptor(const BasisDescriptor<I>& b, const G& graph);
  const BasisDescriptor<I>& basis() const { return basis_;}
private:
  BasisDescriptor<I> basis_;
};

template <class I, class S=std::vector<StateDescriptor<I> > >
class BasisStates : public std::vector<S>
{
public:
  typedef std::vector<S> base_type;
  typedef typename base_type::const_iterator const_iterator;
  typedef S value_type;
  typedef typename base_type::size_type size_type;
  BasisStates(const BasisStatesDescriptor<I>& b);
  size_type index(const value_type& x) const;
  bool check_sort() const;
};

// -------------------------- implementation -----------------------------------

template <class I, class S>
bool BasisStates<I,S>::check_sort() const
{
  for (int i=0;i<size()-1;++i)
    if ((*this)[i]>=(*this)[i+1])
      return false;
  return true;
}

template <class I, class S>
BasisStates<I,S>::BasisStates(const BasisStatesDescriptor<I>& b)
{
  std::vector<int> idx(b.size(),0);
  if (b.size())
  while (true) {
    int k=idx.size()-1;
    while (idx[k]>=b[k].size()) {
      if (b[k].size()==0)
        boost::throw_exception(std::runtime_error("No states for site basis " + 
	     boost::lexical_cast<std::string, SiteBasisDescriptor<I> >(b[k].basis())));
      idx[k]=0;
      --k;
      if (k<0)
        return;
      else
        ++idx[k];
    }
    value_type v;
    for (int i=0;i<idx.size();++i) 
      v.push_back(b[i][idx[i]]);
    push_back(v);
    idx[idx.size()-1]++;
  }
  if (!check_sort())
    boost::throw_exception(std::logic_error("Basis not sorted correctly"));
}


template <class I, class S>
typename BasisStates<I,S>::size_type BasisStates<I,S>::index(const typename BasisStates<I,S>::value_type& x) const
{
  if (binary_search(begin(),end(),x))
    return lower_bound(begin(),end(),x)-begin();
  else
    return size();
  //return std::find(begin(),end(),x)-begin();
}


template <class I> template <class G>
BasisStatesDescriptor<I>::BasisStatesDescriptor(const BasisDescriptor<I>& b, const G& g)
 : basis_(b)
{
  // construct SiteBasisStates for each site
  typename property_map<site_type_t,const G,int>::type site_type(get_or_default(site_type_t(),g,0));
  for (typename boost::graph_traits<G>::vertex_iterator it=sites(g).first;it!=sites(g).second ; ++it) {
    push_back(SiteBasisStates<I>(basis_.site_basis(site_type[*it])));
  }
}


template <class I>
bool BasisDescriptor<I>::set_parameters(const Parameters& p)
{
  bool valid=true;
  for (iterator it=begin();it!=end();++it)
    valid = valid && it->set_parameters(p);
  return valid;
}


template <class I>
const SiteBasisDescriptor<I>& BasisDescriptor<I>::site_basis(int type) const {
  const_iterator it;
  for (it=begin();it!=end();++it)
    if (it->match_type(type))
      break;
  if (it==end())
    boost::throw_exception(std::runtime_error("No matching site basis found for site type" + 
                            boost::lexical_cast<std::string,int>(type) + "\n"));
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
      static_cast<base_type&>(*this) = bases_.find(sitebasis_name_)->second;
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
