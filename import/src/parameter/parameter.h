/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_PARAMETER_PARAMETER_H
#define ALPS_PARAMETER_PARAMETER_H

// for MSVC
#if defined(_MSC_VER)
# pragma warning(disable:4251)
#endif

#include <alps/config.h>
#include <alps/osiris/dump.h>
#include <alps/osiris/std/string.h>
#include <alps/parser/xmlstream.h>
#include <alps/stringvalue.h>
#include <string>

/// \file parameters.h
/// \brief classes to store simulation parameters

namespace alps {

/// \brief a class to store a single parameter value
///
/// the parameter name (key) is stored as a std::string
/// the parameter value is stores as a StringValue.
class ALPS_DECL Parameter
{
public:
  /// the parameter name (key) is stored as a std::string
  typedef std::string key_type;
/// the parameter value is stores as a StringValue.
  typedef StringValue value_type;

  /// deault constructor: no name and no value
  Parameter() : key_(), value_() {}
/// \brief a parameter with a name and value.
  ///
  /// Arbitrary types can be stored. The StringValue constructor will convert
  /// them to a string using boost::lexical_cast
  template<class U>
  Parameter(const key_type& k, const U& v) : key_(k), value_(v) {}

  /// read parameter from a string
  Parameter(std::string const& str) { parse(str); }

  /// read parameter from a string
  void parse(std::string const& str, bool replace_env = true);

  /// replace '${FOO}' with the the content of environment variable FOO
  void replace_envvar();

  /// returns the key (parameter name)
  key_type& key() { return key_; }
  /// returns the key (parameter name)
  const key_type& key() const { return key_; }
  /// returns the value
  value_type& value() { return value_; }
  /// returns the value
  const value_type& value() const { return value_; }

  /// support for Boost serialization
  template<class Archive>
  void serialize(Archive& ar, const unsigned int) {
    std::string v = static_cast<std::string>(value_);
    ar & key_ & v;
    value_ = v;
  }
  
private:
  key_type key_;
  value_type value_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// write parameter in text-form to a std::ostream
ALPS_DECL std::ostream& operator<<(std::ostream& os, const alps::Parameter& p);

//
// OSIRIS support
//

/// ALPS serialization of a parameter value
inline alps::ODump& operator<<(alps::ODump& od, const alps::Parameter& p) {
  return od << p.key() << static_cast<std::string>(p.value());
}

/// ALPS de-serialization of a parameter value
inline alps::IDump& operator>>(alps::IDump& id, alps::Parameter& p) {
  std::string k, v;
  id >> k >> v;
  p = alps::Parameter(k, v);
  return id;
}

/// \brief XML output of a parameter value
///
/// follows the schema on http://xml.comp-phys.org/
inline alps::oxstream& operator<<(alps::oxstream& oxs, const alps::Parameter& parameter)
{
  oxs << alps::start_tag("PARAMETER")
      << alps::attribute("name", parameter.key()) << alps::no_linebreak
      << parameter.value().c_str()
      << alps::end_tag("PARAMETER");
  return oxs;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // ALPS_PARSER_PARAMETER_H
