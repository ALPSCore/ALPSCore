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

#include <alps/model/bondterm.h>
#include <alps/model/operatorsubstitution.h>

#ifndef ALPS_WITHOUT_XML

void alps::BondOperator::read_xml(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  source_ = tag.attributes["source"]=="" ? std::string("i") : tag.attributes["source"];
  target_ = tag.attributes["target"]=="" ? std::string("j") : tag.attributes["target"];
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

void alps::BondOperator::substitute_operators(const ModelLibrary& m, const Parameters& p)
{
  std::vector<std::string> s(2);
  s[0]=source();
  s[1]=target();
  OperatorSubstitution<std::complex<double> > subs(m,p,s);
  Expression e(term());
  e.partial_evaluate(subs);
  e.simplify();
  term_=boost::lexical_cast<std::string>(e);
}

alps::BondTermDescriptor::BondTermDescriptor(const XMLTag& intag, std::istream& is)
{
  XMLTag tag(intag);
  type_ = tag.attributes["type"]=="" ? -1 : boost::lexical_cast<int,std::string>(tag.attributes["type"]);
  read_xml(intag,is);
}

void alps::BondOperator::write_xml(oxstream& os) const
{
  os << start_tag("BONDOPERATOR");
  if (!name().empty())
    os << attribute("name", name());
  if (term()!="")
    os << attribute("source", source()) << attribute("target", target());
  for (Parameters::const_iterator it=parms().begin();it!=parms().end();++it)
    os << start_tag("PARAMETER") << attribute("name", it->key())
       << attribute("default", it->value()) << end_tag("PARAMETER");
  os << term();
  os << end_tag("BONDOPERATOR");
}

void alps::BondTermDescriptor::write_xml(oxstream& os) const
{
  os << start_tag("BONDTERM");
  if (type_>=0)
    os << attribute("type", type_);
  if (term()!="")
    os << attribute("source", source()) << attribute("target", target());
  for (Parameters::const_iterator it=parms().begin();it!=parms().end();++it)
    os << start_tag("PARAMETER") << attribute("name", it->key())
       << attribute("default", it->value()) << end_tag("PARAMETER");
  os << term();
  os << end_tag("BONDTERM");
}

std::set<std::string> alps::BondOperator::operator_names(const Parameters& p) const
{
  std::set<std::string> names;
  typedef std::vector<boost::tuple<Term,SiteOperator,SiteOperator> > V;
  V vec=split(p);
  for (V::const_iterator it=vec.begin();it!=vec.end();++it) {
    std::set<std::string> newnames = boost::tuples::get<1>(*it).operator_names();
    names.insert(newnames.begin(),newnames.end());
    newnames = boost::tuples::get<2>(*it).operator_names();
    names.insert(newnames.begin(),newnames.end());
  }
  return names;
}
 
#endif
