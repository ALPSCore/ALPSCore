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

#include <alps/model/globaloperator.h>
#include <boost/foreach.hpp>

void alps::GlobalOperator::substitute_operators(const ModelLibrary& m, const Parameters& p)
{
  for (std::vector<SiteTermDescriptor>::iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->substitute_operators(m,p);
  for (std::vector<BondTermDescriptor>::iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->substitute_operators(m,p);
  default_siteterm_.substitute_operators(m,p);
  default_bondterm_.substitute_operators(m,p);
}

alps::SiteOperator alps::GlobalOperator::site_term(unsigned int type) const
{
  for (std::vector<SiteTermDescriptor>::const_iterator it =siteterms_.begin();it!=siteterms_.end();++it)
    if (it->match_type(type))
      return it->site_operator();
  return default_siteterm_.get(type).site_operator();
}

alps::BondOperator alps::GlobalOperator::bond_term(unsigned int type) const
{
  for (std::vector<BondTermDescriptor>::const_iterator it =bondterms_.begin();it!=bondterms_.end();++it)
    if (it->match_type(type))
      return it->bond_operator();
  return default_bondterm_.get(type).bond_operator();
}

boost::optional<alps::Parameters> alps::GlobalOperator::create_site_term(unsigned int type)
{
  for (std::vector<SiteTermDescriptor>::const_iterator it =siteterms_.begin();it!=siteterms_.end();++it)
    if (it->match_type(type))
      return boost::optional<Parameters>();
  siteterms_.push_back(default_siteterm_.get(type));
  return default_siteterm_.parms(type);
}

boost::optional<alps::Parameters> alps::GlobalOperator::create_bond_term(unsigned int type)
{
  for (std::vector<BondTermDescriptor>::const_iterator it =bondterms_.begin();it!=bondterms_.end();++it)
    if (it->match_type(type))
      return boost::optional<Parameters>();
  bondterms_.push_back(default_bondterm_.get(type));
  return default_bondterm_.parms(type);
}

alps::Parameters alps::GlobalOperator::create_site_terms(std::set<unsigned int> const& types)
{
  Parameters p;
  BOOST_FOREACH(unsigned int const& t, types) {
    boost::optional<Parameters> newp = create_site_term(t);
    if (newp)
      p << newp.get();
  }
  return p;
}

alps::Parameters alps::GlobalOperator::create_bond_terms(std::set<unsigned int> const& types)
{
  Parameters p;
  BOOST_FOREACH(unsigned int const& t, types) {
    boost::optional<Parameters> newp = create_bond_term(t);
    if (newp)
      p << newp.get();
  }
  return p;
}



#ifndef ALPS_WITHOUT_XML
alps::GlobalOperator::GlobalOperator(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  name_=tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    tag = parse_tag(is);
    tag = read_xml(tag,is);
    if (tag.name != "/"+intag.name)
      boost::throw_exception(std::runtime_error("Encountered illegal tag <"+tag.name+"> in element <"+intag.name+">"));
  }
}

    
alps::XMLTag alps::GlobalOperator::read_xml(const XMLTag& intag, std::istream& is) {
  XMLTag tag(intag);
  while (true) {
    if (tag.name=="SITETERM") {
      if (tag.attributes["type"]=="")
        default_siteterm_=DefaultSiteTermDescriptor(tag,is);
      else
        siteterms_.push_back(SiteTermDescriptor(tag,is));
    }
    else if (tag.name=="BONDTERM") {
      if (tag.attributes["type"]=="")
        default_bondterm_=DefaultBondTermDescriptor(tag,is);
      else
        bondterms_.push_back(BondTermDescriptor(tag,is));
    }
    else
      return tag;
    tag=parse_tag(is);
  }
}

void alps::GlobalOperator::write_xml(oxstream& os) const
{
  os << start_tag("GLOBALOPERATOR") << attribute("name", name());
  write_operators_xml(os);
  os << end_tag("GLOBALOPERATOR");
}

void alps::GlobalOperator::write_operators_xml(oxstream& os) const
{
  if (!default_siteterm_.term().empty())
    default_siteterm_.write_xml(os);
  for (std::vector<SiteTermDescriptor>::const_iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->write_xml(os);
  if (!default_bondterm_.term().empty())
    default_bondterm_.write_xml(os);
  for (std::vector<BondTermDescriptor>::const_iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->write_xml(os);
  
}

#endif
