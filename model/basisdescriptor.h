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

#ifndef ALPS_MODEL_BASISDESCRIPTOR_H
#define ALPS_MODEL_BASISDESCRIPTOR_H

#include <alps/model/sitebasisstates.h>
#include <alps/lattice/lattice.h>
#include <vector>

namespace alps {

template<class I>
class site_basis_match : public SiteBasisDescriptor<I>
{
public:
  typedef SiteBasisDescriptor<I> super_type;
  typedef typename super_type::const_iterator const_iterator;
  typedef std::map<std::string,SiteBasisDescriptor<I> > sitebasis_map_type;

  site_basis_match() : type_(-2) {} // matches no site type
  site_basis_match(const super_type& site_basis, int type = -2)
    : super_type(site_basis), type_(type), sitebasis_name_() {}
  site_basis_match(const std::string& name, int type = -2)
    : super_type(), type_(type), sitebasis_name_(name) {}
  site_basis_match(const XMLTag&, std::istream&,
                 const sitebasis_map_type& bases_= sitebasis_map_type());

  void write_xml(oxstream&) const;
  bool match_type(int type) const { return type_==-1 || type==type_;}

private:
  int type_;
  std::string sitebasis_name_;
  Parameters parms_;
};


template<class I>
class BasisDescriptor : public std::vector<site_basis_match<I> >
{
  typedef std::vector<site_basis_match<I> > super_type;
public:
  typedef typename super_type::iterator iterator;
  typedef typename super_type::const_iterator const_iterator;
  typedef std::map<std::string,SiteBasisDescriptor<I> > sitebasis_map_type;
  typedef std::vector<std::pair<std::string,half_integer<I> > > constraints_type;
  typedef alps::Expression expression_type;
  typedef std::vector<std::pair<std::string, expression_type> > unevaluated_constraints_type;

  BasisDescriptor(const std::string& name = "") : name_(name) {}
  BasisDescriptor(const XMLTag&, std::istream&,
                  const sitebasis_map_type& bases_= sitebasis_map_type());

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
  for (iterator it=super_type::begin();it!=super_type::end();++it)
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
#ifndef ALPS_WITH_NEW_EXPRESSION
      evaluated_constraints_.push_back(std::make_pair(it->first,half_integer<I>(alps::evaluate(it->second, p))));
#else
      evaluated_constraints_.push_back(std::make_pair(it->first,half_integer<I>(alps::evaluate<double>(it->second, p))));
#endif
    else
      unevaluated_constraints_.push_back(*it);
}


template <class I>
const SiteBasisDescriptor<I>& BasisDescriptor<I>::site_basis(int type) const {
  const_iterator it;
  for (it=super_type::begin();it!=super_type::end();++it)
    if (it->match_type(type))
      break;
  if (it==super_type::end())
    boost::throw_exception(std::runtime_error("No matching site basis found for site type" +
                            boost::lexical_cast<std::string,int>(type) + "\n"));
  return *it;
}


template <class I>
site_basis_match<I>::site_basis_match(const XMLTag& intag, std::istream& is, const sitebasis_map_type& bases_)
{
  XMLTag tag(intag);
  sitebasis_name_ = tag.attributes["ref"];
  type_ = tag.attributes["type"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["type"]);
  if (sitebasis_name_=="") {
    super_type sitebasis(intag,is);
    std::copy(sitebasis.begin(),sitebasis.end(),std::back_inserter(*this));
  }
  else {
    if (bases_.find(sitebasis_name_)==bases_.end())
      boost::throw_exception(std::runtime_error("unknown site basis: " + sitebasis_name_ + " in <BASIS>"));
    else
      static_cast<super_type&>(*this) = bases_.find(sitebasis_name_)->second;
    if (tag.type!=XMLTag::SINGLE) {
      tag = parse_tag(is);
      while (tag.name=="PARAMETER") {
        parms_[tag.attributes["name"]]=tag.attributes["value"];
        if (tag.type!=XMLTag::SINGLE)
          tag = parse_tag(is);
        tag = parse_tag(is);
      }
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
      push_back(site_basis_match<I>(tag,is,bases_));
      tag = parse_tag(is);
    }
    while (tag.name=="CONSTRAINT") {
      constraints_.push_back(std::make_pair(tag.attributes["quantumnumber"], expression_type(tag.attributes["value"])));
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
void site_basis_match<I>::write_xml(oxstream& os) const
{
  os << start_tag("SITEBASIS");
  if (type_>=0)
    os << attribute("type", type_);
  if (sitebasis_name_!="") {
    os << attribute("ref", sitebasis_name_);
    for (Parameters::const_iterator it = parms_.begin(); it != parms_.end(); ++it)
      os << start_tag("PARAMETER") << attribute("name", it->key())
         << attribute("value", it->value()) << end_tag("PARAMETER");
  } else {
    for (Parameters::const_iterator p_itr = super_type::get_parameters().begin();
         p_itr != super_type::get_parameters().end(); ++p_itr)
      os << start_tag("PARAMETER") << attribute("name", p_itr->key())
         << attribute("default", p_itr->value()) << end_tag("PARAMETER");
    for (const_iterator it = super_type::begin(); it != super_type::end(); ++it)
      os << *it;
  }
  os << end_tag("SITEBASIS");
}

template <class I>
void BasisDescriptor<I>::write_xml(oxstream& os) const
{
  os << start_tag("BASIS") << attribute("name", name());
  for (const_iterator it=super_type::begin();it!=super_type::end();++it)
    os << *it;
  for (typename unevaluated_constraints_type::const_iterator
         it=constraints_.begin(); it!=constraints_.end(); ++it)
    os << start_tag("CONSTRAINT") << attribute("quantumnumber",it->first)
       << attribute("value", boost::lexical_cast<std::string>(it->second))
       << end_tag("CONSTRAINT");
  os << end_tag("BASIS");
}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline alps::oxstream& operator<<(alps::oxstream& out, const alps::site_basis_match<I>& q)
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
inline std::ostream& operator<<(std::ostream& out, const alps::site_basis_match<I>& q)
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


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // namespace alps
#endif

#endif
