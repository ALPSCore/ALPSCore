/***************************************************************************
* ALPS++ library
*
* alps/parser/parameterlist.C   An array of parameters
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

#include <alps/parameterlist.h>
#include <alps/parser/parser.h>

#include <boost/throw_exception.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>

namespace alps {

void ParameterList::parse(std::istream& is)
{
  Parameters global;
  char c;
  while (true) {
    is >> global;
    is >> c;
    if (!is) break;
    if(c=='{') {
      // new block starts with {
      // make new Parameters as clone of global
      push_back(global);
      is >> *rbegin();
      check_character(is,'}',"} expected in parameter list");
    } else {
      is.putback(c);
      break;
    }
  }
}

} // namespace alps

//
// XML support
//

#ifndef ALPS_WITHOUT_XML

namespace alps {

ParameterListXMLHandler::ParameterListXMLHandler(ParameterList& list)
  : CompositeXMLHandler("PARAMETERLIST"), list_(list),
    parameter_(), default_(), current_(), parameter_handler_(parameter_), 
    current_handler_(current_) {
  add_handler(parameter_handler_);
  add_handler(current_handler_);
}

void ParameterListXMLHandler::start_child(const std::string& name,
  const XMLAttributes& /* attributes */) {
  if (name == "PARAMETER") {
    parameter_ = Parameter();
  } else if (name == "PARAMETERS") {
    current_ = default_;
  }
}
void ParameterListXMLHandler::end_child(const std::string& name) {
  if (name == "PARAMETER") {
    default_[parameter_.key()] = parameter_.value();
  } else if (name == "PARAMETERS") {
    list_.push_back(current_);
  }
}

} // namespace alps

#endif
