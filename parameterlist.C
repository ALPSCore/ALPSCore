/***************************************************************************
* ALPS++ library
*
* alps/parser/parameterlist.C   An array of parameters
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This software is part of the ALPS library, published under the 
* ALPS Library License; you can use, redistribute it and/or modify 
* it under the terms of the License, either version 1 or (at your option) 
* any later version.
*
* You should have received a copy of the ALPS Library License along with 
* the ALPS Library; see the file License.txt. If not, the license is also 
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
