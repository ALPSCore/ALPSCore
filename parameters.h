/***************************************************************************
* ALPS++ library
*
* parser/parameters.h
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_PARSER_PARAMETERS_H
#define ALPS_PARSER_PARAMETERS_H

#include <alps/config.h>
#include <alps/stringvalue.h>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif
#ifndef ALPS_WITHOUT_XML
# include <alps/parser/parser.h>
# include <alps/xml.h>
#endif

#include <boost/throw_exception.hpp>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace alps {

class Parameter
{
public:
  typedef std::string key_type;
  typedef StringValue value_type;

  Parameter() : key_(), value_() {}
  Parameter(const Parameter& p) : key_(p.key_), value_(p.value_) {}
  Parameter(const key_type& k) : key_(k), value_() {}
  Parameter(const key_type& k, const value_type& v) : key_(k), value_(v) {}
  Parameter(const key_type& k, const char * v) : key_(k), value_(v) {}
  template<class U>
  Parameter(const key_type& k, const U& v) : key_(k), value_(v) {}

  key_type& key() { return key_; }
  const key_type& key() const { return key_; }
  value_type& value() { return value_; }
  const value_type& value() const { return value_; }

private:
  key_type key_;
  value_type value_;
};

class Parameters
{
public:
  typedef std::string                     key_type;
  typedef StringValue                     value_type;
  typedef Parameter                       parameter_type;

  typedef std::vector<parameter_type>     list_type;
  typedef list_type::size_type            size_type;

  typedef std::map<key_type, size_type>   map_type;

  typedef parameter_type *            pointer_type;
  typedef const parameter_type *      const_pointer_type;
  typedef parameter_type &            reference_type;
  typedef const parameter_type &      const_reference_type;
  typedef list_type::iterator         iterator;
  typedef list_type::const_iterator   const_iterator;

  Parameters() {}
  Parameters(std::istream& is) { parse(is); }

  void parse(std::istream& is);

  void clear() { list_.clear(); map_.clear(); }
  size_type size() const { return list_.size(); }

  bool defined(const key_type& k) const {
    return (map_.find(k) != map_.end());
  }

  // accessing elements by key
  value_type& operator[](const key_type& k) {
    if (defined(k)) {
      return list_[map_.find(k)->second].value();
    } else {
      push_back(k, value_type());
      return list_.rbegin()->value();
    }
  }
  const value_type& operator[](const key_type& k) const {
    if (!defined(k))
      boost::throw_exception(std::runtime_error("parameter not defined"));
    return list_[map_.find(k)->second].value();
  }
  value_type value_or_default(const key_type& k, const value_type& v) const {
    return defined(k) ? (*this)[k] : v;
  }

  iterator begin() { return list_.begin(); }
  const_iterator begin() const { return list_.begin(); }
  iterator end() { return list_.end(); }
  const_iterator end() const { return list_.end(); }

  void push_back(const parameter_type& p) {
    if (p.key().empty())
      boost::throw_exception(std::runtime_error("empty key"));
    if (defined(p.key()))
      boost::throw_exception(std::runtime_error("duplicated parameter: " + p.key()));
    map_[p.key()] = list_.size();
    list_.push_back(p);
  }
  void push_back(const key_type& k, const value_type& v) {
    push_back(Parameter(k, v));
  }
  Parameters& operator<<(const parameter_type& p) {
    (*this)[p.key()] = p.value();
    return *this;
  }
  Parameters& operator<<(const Parameters& params) {
    for (const_iterator it = params.begin(); it != params.end(); ++it)
      (*this) << *it;
    return *this;
  }

  void write_xml(std::ostream& xml) const
  {
    xml << "<PARAMETERS>\n";
    for (const_iterator it = begin(); it != end(); ++it) {
      if (it->value().valid())
	xml << "<PARAMETER name=\"" << it->key() << "\">" << it->value()
	    << "</PARAMETER>\n";
      xml << "</PARAMETERS>\n";
    }
  }

  void read_xml(XMLTag tag, std::istream& xml)
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
      push_back(name, parse_content(xml));
      tag = parse_tag(xml);
      if(tag.name!="/PARAMETER")
	boost::throw_exception(std::runtime_error("</PARAMETER> expected at end of <PARAMETER> element"));
      tag = parse_tag(xml);
    }
  }

private:
  list_type list_;
  map_type map_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& os, const alps::Parameters& p)
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

inline std::istream& operator>>(std::istream& is, alps::Parameters& p)
{
  p.parse(is);
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif


//
// OSIRIS support
//

#ifndef ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::ODump& operator<<(alps::ODump& od, const alps::Parameter& p)
{ return od << p.key() << static_cast<std::string>(p.value()); }

inline alps::IDump& operator>>(alps::IDump& id, alps::Parameter& p)
{
  std::string k, v;
  id >> k >> v;
  p = alps::Parameter(k, v);
  return id;
}

inline alps::ODump& operator<<(alps::ODump& od, const alps::Parameters& p)
{
  od << uint32_t(p.size());
  for (alps::Parameters::const_iterator it = p.begin(); it != p.end(); ++it)
    od << *it;
  return od;
}

inline alps::IDump& operator>>(alps::IDump& id, alps::Parameters& p)
{
  p.clear();
  uint32_t n(id);
  for (std::size_t i = 0; i < n; ++i) {
    Parameter m;
    id >> m;
    p.push_back(m);
  }
  return id;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // !ALPS_WITHOUT_OSIRIS


//
// XML support
//

#ifndef ALPS_WITHOUT_XML

namespace alps {

class ParameterXMLHandler : public XMLHandlerBase
{
public:
  ParameterXMLHandler(Parameter& p);
  
  void start_element(const std::string& name,
                     const XMLAttributes& attributes);
  void end_element(const std::string& name);
  void text(const std::string& text);
  
private:
  Parameter& parameter_;
};

class ParametersXMLHandler : public CompositeXMLHandler
{
public:
  ParametersXMLHandler(Parameters& p);

protected:  
  void start_child(const std::string& name,
		   const XMLAttributes& attributes);
  void end_child(const std::string& name);
  
private:
  Parameters& parameters_;
  Parameter parameter_;
  ParameterXMLHandler handler_;
};

} // namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::oxstream& operator<<(alps::oxstream& oxs,
				  const alps::Parameter& parameter)
{
  oxs << alps::start_tag("PARAMETER")
      << alps::attribute("name", parameter.key()) << alps::no_linebreak
      << parameter.value().c_str()
      << alps::end_tag("PARAMETER");
  return oxs;
}

inline alps::oxstream& operator<<(alps::oxstream& oxs,
				  const alps::Parameters& parameters)
{
  oxs << alps::start_tag("PARAMETERS");
  alps::Parameters::const_iterator p_end = parameters.end();
  for (alps::Parameters::const_iterator p = parameters.begin(); p != p_end;
       ++p) oxs << *p;
  oxs << alps::end_tag("PARAMETERS");
  return oxs;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // !ALPS_WITHOUT_XML

#endif // ALPS_PARSER_PARAMETERS_H
