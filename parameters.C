/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2004 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/parameters.h>
#include <alps/cctype.h>
#include <alps/parser/parser.h>

#include <boost/throw_exception.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <stdlib.h>

namespace alps {

void Parameters::push_back(const parameter_type& p, bool allow_overwrite)
{
  if (p.key().empty())
    boost::throw_exception(std::runtime_error("empty key"));
  if (defined(p.key())) {
    if (allow_overwrite)
      list_[map_.find(p.key())->second].value()=p.value();
    else
      boost::throw_exception(std::runtime_error("duplicated parameter: " + p.key()));
  }
  else {
    map_[p.key()] = list_.size();
    list_.push_back(p);
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

void Parameters::parse(std::istream& is)
{
  char c;
  do {
    is >> c;
    while (is && (c==';' || c==',')) is >> c;  // ignore extra semi-colons
    if (!is) break;
    if (std::isalpha(c)) {
      is.putback(c);
      std::string key = parse_parameter_name(is);
      std::string value;

      check_character(is, '=',
        "= expected in assignment while parsing Parameters");

      is >> c;
      switch (c) {
      case '[':
        value = read_until(is, ']');
        break;
      case '"':
        value = read_until(is, '"');
        break;
      case '\'':
        value = read_until(is, '\'');
        break;
      case '$':
        check_character(is, '{', "{ expected in Parameter environment variable expansion");
        value = read_until(is, '}');
	{
	  char const* EnvStr = getenv(value.c_str());
	  if (EnvStr)
	    value = EnvStr;  // if the environment string exists, then substitute its value
	  else
	    value = "${" + value + '}'; // pass through unchanged if the environment string doesnt exist
	}
        break;

      default:
        while(c!=';' && c!=',' && c!='}' && c!= '{' &&
              c!='\r' && c!='\n' && is) {
          value+=c;
          c = is.get();
        }
        if (c=='{' || c=='}')
          is.putback(c);
      }
      push_back(key, value, true);
    } else {
      is.putback(c);
      break;
    }
  } while (true);
}


//
// XML support
//

ParameterXMLHandler::ParameterXMLHandler(Parameter& p)
  : XMLHandlerBase("PARAMETER"), parameter_(p) {}

void ParameterXMLHandler::start_element(const std::string& name,
                                        const XMLAttributes& attributes)
{
  if (name != "PARAMETER")
    boost::throw_exception(std::runtime_error("ParameterXMLHandler: unknown tag name : " + name));
  if (!attributes.defined("name"))
    boost::throw_exception(std::runtime_error("ParameterXMLHandler: name attribute not found in PARAMETER tag"));
  parameter_.key() = attributes["name"];
}

void ParameterXMLHandler::end_element(const std::string&) {}

void ParameterXMLHandler::text(const std::string& text) {
  parameter_.value() = text;
}

ParametersXMLHandler::ParametersXMLHandler(Parameters& p)
  : CompositeXMLHandler("PARAMETERS"), parameters_(p), parameter_(),
    handler_(parameter_)
{
  add_handler(handler_);
}

void ParametersXMLHandler::start_child(const std::string&,
                                       const XMLAttributes&)
{
  parameter_ = Parameter();
}

void ParametersXMLHandler::end_child(const std::string&)
{
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
