/***************************************************************************
* ALPS++ library
*
* alps/parser/parameters.C
*
* $Id$
*
* Copyright (C) 1994-2002 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
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

#include <alps/parameters.h>
#include <alps/expression.h>
#include <alps/cctype.h>
#include <alps/parser/parser.h>

#include <boost/throw_exception.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace alps {

void Parameters::parse(std::istream& is)
{
  char c;
  do {   
    is >> c;
    if (!is)
      break;
    while (c==';' || c==',') is >> c;  // ignore extra semi-colons  
    if (is && std::isalpha(c)) {
      is.putback(c);
      std::string key = parse_identifier(is);
      std::string value;

      check_character(is, '=', "= expected in assignment while parsing Parameters");

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
      default:                            
	while(c!=';' && c!=',' && c!='}' && c!= '{' && c!='\r' && c!='\n' && is) {
	  value+=c;
	  c = is.get();
	}
	if (c=='{' || c=='}')
	  is.putback(c);
      }
      push_back(key, value);
    } else {
      is.putback(c);
      break;
    }
  } while (true);
}

} // namespace alps

//
// XML support
//

#ifndef ALPS_WITHOUT_XML

namespace alps {

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

#endif // !ALPS_WITHOUT_XML
