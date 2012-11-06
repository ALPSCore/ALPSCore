/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2008 by Matthias Troyer <troyer@comp-phys.org>,
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

#include "parameters.h"
#include "parameters_p.h"
#include <boost/foreach.hpp>
#include <cstdlib>
#include <iostream>
#include <streambuf>

#include <alps/expression.h>

#include <alps/hdf5.hpp>

namespace bs = boost::spirit;

namespace alps {

void Parameters::push_back(const parameter_type& p, bool allow_overwrite)
{
  if (p.key().empty())
    boost::throw_exception(std::runtime_error("empty key"));
  if (defined(p.key())) {
    if (allow_overwrite)
      map_.find(p.key())->second->value() = p.value();
    else
      boost::throw_exception(std::runtime_error("duplicated parameter: " + p.key()));
  } else {
    list_.push_back(p);
    map_[p.key()] = --list_.end();
  }
}

Parameters& Parameters::operator<<(const Parameters& params)
{
  for (const_iterator it = params.begin(); it != params.end(); ++it)
    (*this) << *it;
  return *this;
}

void Parameters::copy_undefined(const Parameters& p)
{
  for (const_iterator it=p.begin();it!=p.end();++it)
    if (!defined(it->key()))
      push_back(*it);
}

void Parameters::read_xml(XMLTag tag, std::istream& xml,bool ignore_duplicates)
{
    if (tag.name!="PARAMETERS")
      boost::throw_exception(std::runtime_error("<PARAMETERS> element expected"));
    if (tag.type==XMLTag::SINGLE)
      return;
    tag = parse_tag(xml);
    while (tag.name!="/PARAMETERS") {
      if(tag.name!="PARAMETER")
        boost::throw_exception(std::runtime_error("<PARAMETER> element expected in <PARAMETERS>"));
      std::string name = tag.attributes["name"];
      if(name=="")
        boost::throw_exception(std::runtime_error("nonempty name attribute expected in <PARAMETER>"));
      push_back(name, parse_content(xml),ignore_duplicates);
      tag = parse_tag(xml);
      if(tag.name!="/PARAMETER")
        boost::throw_exception(std::runtime_error("</PARAMETER> expected at end of <PARAMETER> element"));
      tag = parse_tag(xml);
    }
}

void Parameters::extract_from_xml(std::istream& infile)
{
  XMLTag tag=alps::parse_tag(infile,true);
  std::string closingtag = "/"+tag.name;
  tag=parse_tag(infile,true);
  while (tag.name!="PARAMETERS" && tag.name != closingtag) {
    skip_element(infile,tag);
    tag=parse_tag(infile,true);
  }
  read_xml(tag,infile);
}

void Parameters::parse(std::istream& is, bool replace_env) {
  std::deque<char> buff;
  std::copy(std::istreambuf_iterator<char>(is), std::istreambuf_iterator<char>(),
    std::back_inserter(buff));
  bs::parse_info<std::deque<char>::iterator> info = bs::parse(
    buff.begin(), buff.end(),
    ParametersParser(*this) >> bs::end_p,
    bs::blank_p | bs::comment_p("//") | bs::comment_p("/*", "*/"));

  /* // ST 2006.10.06: following in-situ version does not work with Intel C++ on IA64
  typedef bs::multi_pass<std::istreambuf_iterator<char> > iterator_t;
  iterator_t first = bs::make_multi_pass(std::istreambuf_iterator<char>(is));
  iterator_t last = bs::make_multi_pass(std::istreambuf_iterator<char>());
  bs::parse_info<iterator_t> info = bs::parse(
    first, last,
    ...
  */

  if (!info.full) {
    std::deque<char>::iterator itr = info.stop;
    std::string err = "parameter parse error at \"";
    for (int i = 0; itr != buff.end() && i < 32; ++itr, ++i)
      err += (*itr != '\n' ? *itr : ' ');
    boost::throw_exception(std::runtime_error(err + "\""));
  }
  if (replace_env) replace_envvar();
}

void Parameters::replace_envvar() {
  BOOST_FOREACH(Parameter& p, list_) p.replace_envvar();
}

void Parameters::save(hdf5::archive & ar) const {
    expression::ParameterEvaluator<double> eval(*this,false);
    for (const_iterator it = begin(); it != end(); ++it) {
        try {
            expression::Expression<double> expr(it->value());
            if (expr.can_evaluate(eval)) {
                double value = expr.value(eval);
                if (numeric::is_zero(value - static_cast<double>(static_cast<int>(value)))) 
                    ar << make_pvp(it->key(), static_cast<int>(value+ (value > 0 ? 0.25 : -0.25)));
                else 
                    ar << make_pvp(it->key(), value);
            } else {
                expr.partial_evaluate(eval);
                ar << make_pvp(it->key(), boost::lexical_cast<std::string>(expr));
            }
        } catch(...) {
          // we had a problem evaluating, use original full value
          ar << make_pvp(it->key(), boost::lexical_cast<std::string>(it->value()));
        }
    }
}
void Parameters::load(hdf5::archive & ar) {
  std::vector<std::string> list = ar.list_children(ar.get_context());

  for (std::vector<std::string>::const_iterator it = list.begin(); it != list.end(); ++it) {
    std::string v;
    ar >> make_pvp(*it, v);
    operator[](*it) = v;
  }
}

//
// XML support
//

ParametersXMLHandler::ParametersXMLHandler(Parameters& p)
  : CompositeXMLHandler("PARAMETERS"), parameters_(p), parameter_(),
    handler_(parameter_)
{
  add_handler(handler_);
}

void ParametersXMLHandler::start_child(const std::string&,
                                       const XMLAttributes&,
                                       xml::tag_type type)
{ if (type == xml::element) parameter_ = Parameter(); }

void ParametersXMLHandler::end_child(const std::string&, xml::tag_type type)
{
  if (type == xml::element)
    parameters_.operator[](parameter_.key()) = parameter_.value();
}

} // namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

std::ostream& operator<<(std::ostream& os, const alps::Parameters& p)
{
  for (alps::Parameters::const_iterator it = p.begin(); it != p.end(); ++it) {
    if (it->value().valid()) {
      std::string s = it->value().c_str();
      os << it->key() << " = ";
      if (s.find(' ') != std::string::npos)
        os << '"' << s << '"';
      else
        os << s;
      os << ";\n";
    }
  }
  return os;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif
