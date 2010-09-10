/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>
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

#include <alps/model/siteterm.h>
#include <alps/model/operatorsubstitution.h>

// some file (probably a python header) defines a tolower macro ...
#undef tolower
#undef toupper

#include <boost/regex.hpp> 

#ifndef ALPS_WITHOUT_XML

void alps::SiteOperator::read_xml(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  site_ = tag.attributes["site"];
  name_ = tag.attributes["name"];
  if (tag.type!=XMLTag::SINGLE) {
    term_=parse_content(is);
    while (true) {
      tag = parse_tag(is,false);
      if (tag.name == "/"+intag.name)
        return;
      if (tag.name == "PARAMETER") {
        parms_[tag.attributes["name"]]=tag.attributes["default"];
        if (tag.type!=XMLTag::SINGLE) {
          tag=parse_tag(is);
          if (tag.name!="/PARAMETER")
            boost::throw_exception(std::runtime_error("End tag </PARAMETER> missing while parsing " + name() + " Hamiltonian"));
        }
      }
      else if (tag.type!=XMLTag::COMMENT)
        boost::throw_exception(std::runtime_error("Illegal tag <" + tag.name + "> in <" +intag.name+ "> element"));
      std::string next_part = parse_content(is);
      if (!term_.empty() && !next_part.empty())
        term_ += " ";
      term_ +=next_part;
    }
  }
}

void alps::SiteOperator::substitute_operators(const ModelLibrary& m, const Parameters& p)
{
  std::vector<std::string> s(1,site());
  OperatorSubstitution<std::complex<double> > subs(m,p,s);
  Expression e(term());
  e.partial_evaluate(subs);
  e.simplify();
  term_=boost::lexical_cast<std::string>(e);
}


alps::SiteTermDescriptor::SiteTermDescriptor(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  type_ = tag.attributes["type"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["type"]);
  read_xml(intag,is);
}

void alps::SiteOperator::write_xml(oxstream& os) const
{
  os << start_tag("SITEOPERATOR");
  if (!name().empty())
    os << attribute("name", name());
  if (!site().empty())
    os << attribute("site", site());
  for (Parameters::const_iterator it=parms().begin();it!=parms().end();++it)
    os << start_tag("PARAMETER") << attribute("name", it->key())
       << attribute("default", it->value()) << end_tag("PARAMETER");
  os << term() << end_tag("SITEOPERATOR");
}

void alps::SiteTermDescriptor::write_xml(oxstream& os) const
{
  os << start_tag("SITETERM");
  if (type_>=0)
    os << attribute("type", type_);
  if (!site().empty())
    os << attribute("site", site());
  for (Parameters::const_iterator it=parms().begin();it!=parms().end();++it)
    os << start_tag("PARAMETER") << attribute("name", it->key())
       << attribute("default", it->value()) << end_tag("PARAMETER");
  os << term() << end_tag("SITETERM");
}

std::set<std::string> alps::SiteOperator::operator_names() const
{
  std::set<std::string> names;
  boost::regex expression("^(.*)\\(.*\\)$");
  boost::smatch what;
  alps::Expression ex(term_);
  ex.flatten();
  ex.simplify();
  for (Expression::term_iterator tit = ex.terms().first;
       tit != ex.terms().second; ++tit)
    for (Term::factor_iterator fit = tit->factors().first;
         fit != tit->factors().second; ++fit) {
      std::string name = boost::lexical_cast<std::string>(*fit);
      if (boost::regex_match(name, what, expression))
        names.insert(std::string(what[1].first, what[1].second));
    }
  return names;
}
 
#endif
