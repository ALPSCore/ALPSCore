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

void alps::GlobalOperator::substitute_operators(const ModelLibrary& m, const Parameters& p)
{
  for (std::vector<SiteTermDescriptor>::iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->substitute_operators(m,p);
  for (std::vector<BondTermDescriptor>::iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->substitute_operators(m,p);
}

alps::SiteTermDescriptor alps::GlobalOperator::site_term(int type) const
{
  for (std::vector<SiteTermDescriptor>::const_iterator it =siteterms_.begin();it!=siteterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return SiteTermDescriptor();
}

alps::BondTermDescriptor alps::GlobalOperator::bond_term(int type) const
{
  for (std::vector<BondTermDescriptor>::const_iterator it =bondterms_.begin();it!=bondterms_.end();++it)
    if (it->match_type(type))
      return *it;
  return BondTermDescriptor();
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
    if (tag.name=="SITETERM")
      siteterms_.push_back(SiteTermDescriptor(tag,is));
    else if (tag.name=="BONDTERM")
      bondterms_.push_back(BondTermDescriptor(tag,is));
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
  for (std::vector<SiteTermDescriptor>::const_iterator it=siteterms_.begin();it!=siteterms_.end();++it)
    it->write_xml(os);
  for (std::vector<BondTermDescriptor>::const_iterator it=bondterms_.begin();it!=bondterms_.end();++it)
    it->write_xml(os);
}

#endif
