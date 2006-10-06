/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#include "parameter.h"
#include "parameter_p.h"
#include <boost/throw_exception.hpp>
#include <iostream>
#include <stdexcept>
#include <string>
#include <cstdlib>

namespace bs = boost::spirit;

namespace alps {

namespace {

struct insert_helper {
  insert_helper(std::string& str) : str_(str) {}
  template<typename IteratorT>
  void operator()(IteratorT first, IteratorT last) const {
    str_ += std::string(first, last);
  }
  std::string& str_;
};
  
struct replace_helper {
  replace_helper(std::string& str) : str_(str) {}
  template<typename IteratorT>
  void operator()(IteratorT first, IteratorT last) const {
    std::string v(first, last);
    char const* env = std::getenv(v.c_str());
    if (env)
      str_ += env;
    else 
      str_ += "${" + v + "}";
  }
  std::string& str_;
};
  
}

void Parameter::parse(std::string const& str, bool replace_env) {
  if (!bs::parse(str.c_str(),
    ParameterParser(*this) >> !bs::ch_p(';') >> !bs::end_p, bs::blank_p).full)
    boost::throw_exception(std::runtime_error("can not parse '" + str + "'"));
  if (replace_env) replace_envvar();
}
 
void Parameter::replace_envvar() {
  std::string value = value_;
  std::string result;
  insert_helper insert(result);
  replace_helper replace(result);
  if (!bs::parse(value.c_str(),
    *(bs::anychar_p - '$')[insert]
    % ( bs::ch_p('$') >> bs::confix_p('{', (*bs::anychar_p)[replace], '}') )
    >> bs::end_p
    ).full)
    boost::throw_exception(std::runtime_error("can not parse '" + value + "'"));
  value_ = result;
}

//
// XML support
//

ParameterXMLHandler::ParameterXMLHandler(Parameter& p) :
  XMLHandlerBase("PARAMETER"), parameter_(p) {}

void ParameterXMLHandler::start_element(const std::string& name, const XMLAttributes& attributes,
  xml::tag_type type) {
  if (type == xml::element) {
    if (name != "PARAMETER")
      boost::throw_exception(std::runtime_error(
        "ParameterXMLHandler: unknown tag name : " + name));
    if (!attributes.defined("name"))
      boost::throw_exception(std::runtime_error(
        "ParameterXMLHandler: name attribute not found in PARAMETER tag"));
    parameter_.key() = attributes["name"];
  }
}

void ParameterXMLHandler::end_element(const std::string&, xml::tag_type) {}

void ParameterXMLHandler::text(const std::string& text) {
  parameter_.value() = text;
}

} // namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

std::ostream& operator<<(std::ostream& os, const alps::Parameter& p) {
  if (p.value().valid()) {
    std::string s = p.value().c_str();
    os << p.key() << " = ";
    if (s.find(' ') != std::string::npos)
      os << '"' << s << '"';
    else
      os << s;
    os << ";";
  }
  return os;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif
