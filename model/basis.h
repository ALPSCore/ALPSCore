/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003 by Matthias Troyer <troyer@comp-phys.org>
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
  SiteBasisMatch(const XMLTag&, std::istream&,const sitebasis_map_type& bases_= sitebasis_map_type());
  void write_xml(oxstream&) const;
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
  typedef std::vector<std::pair<std::string,half_integer<I> > > constraints_type;
  typedef std::vector<std::pair<std::string,alps::Expression> > unevaluated_constraints_type;

  BasisDescriptor() {}
  BasisDescriptor(const XMLTag&, std::istream&,const sitebasis_map_type& bases_= sitebasis_map_type());
  void write_xml(oxstream&) const;
  const SiteBasisDescriptor<I>& site_basis(int type=0) const;
  const std::string& name() const { return name_;}
  bool set_parameters(const Parameters& p);
  const constraints_type& constraints() const { return evaluated_constraints_;}
  const unevaluated_constraints_type& unevaluated_constraints() const { return unevaluated_constraints_;}
  const unevaluated_constraints_type& all_constraints() const { return constraints_;}
private:
  std::string name_;
  void check_constraints(const Parameters& =Parameters());
  unevaluated_constraints_type constraints_;
  unevaluated_constraints_type unevaluated_constraints_;
  constraints_type evaluated_constraints_;
};


// -------------------------- implementation -----------------------------------


template <class I>
bool BasisDescriptor<I>::set_parameters(const Parameters& p)
{
  bool valid=true;
  for (iterator it=begin();it!=end();++it)
    valid = valid && it->set_parameters(p);
  check_constraints(p);
  return valid;
}

template <class I>
void BasisDescriptor<I>::check_constraints(const Parameters& p)
{
  evaluated_constraints_.clear();
  unevaluated_constraints_.clear();
  for (typename unevaluated_constraints_type::iterator it=constraints_.begin();it!=constraints_.end();++it)
    if (it->second.can_evaluate(p))
      evaluated_constraints_.push_back(std::make_pair(it->first,half_integer<I>(it->second.value(p))));
    else
      unevaluated_constraints_.push_back(*it);
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
    while (tag.name=="CONSTRAINT") {
      constraints_.push_back(std::make_pair(tag.attributes["quantumnumber"],Expression(tag.attributes["value"])));
      if (tag.type!=XMLTag::SINGLE) {
        tag=parse_tag(is);
        if (tag.name!="/CONSTRAINT")
          boost::throw_exception(std::runtime_error("Unexpected tag " + tag.name + " in <CONSTRAINT> element"));
      }
      tag = parse_tag(is);
    }
    if (tag.name !="/BASIS")
      boost::throw_exception(std::runtime_error("Unexpected tag <" + tag.name + "> in <BASIS> element"));
  }
  check_constraints();
}

template <class I>
void SiteBasisMatch<I>::write_xml(oxstream& os) const
{
  os << start_tag("SITEBASIS");
  if (type_>=0) 
    os << attribute("type", type_);
  if (sitebasis_name_!="")
    os << attribute("ref", sitebasis_name_);
  else
    for (const_iterator it=begin();it!=end();++it)
      os << *it;
  os << end_tag("SITEBASIS");
}

template <class I>
void BasisDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("BASIS") << attribute("name", name());
  for (const_iterator it=begin();it!=end();++it)
    os << *it;
  for (typename unevaluated_constraints_type::const_iterator it=constraints_.begin();it!=constraints_.end();++it)
    os << start_tag("CONSTRAINT") << attribute("quantumnumber",it->first) 
       << attribute("value",static_cast<std::string>(it->second)) << end_tag("CONSTRAINT");
  os << end_tag("BASIS");
}

#endif

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

#ifndef ALPS_WITHOUT_XML

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::SiteBasisMatch<I>& q)
{
  q.write_xml(out);
  return out;        
}


template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::BasisDescriptor<I>& q)
{
  q.write_xml(out);
  return out;        
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::SiteBasisMatch<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;        
}

template <class I>
inline std::ostream& operator<<(std::ostream& out, const alps::BasisDescriptor<I>& q)
{
  alps::oxstream xml(out);
  xml << q;
  return out;        
}


#endif

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
