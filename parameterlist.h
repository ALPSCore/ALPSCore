/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_PARSER_PARAMETERLIST_H
#define ALPS_PARSER_PARAMETERLIST_H

#include <alps/config.h>
#include <alps/parameters.h>
#include <boost/serialization/vector.hpp>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif
#ifndef ALPS_WITHOUT_XML
# include <alps/xml.h>
#endif

#include <iostream>
#include <vector>

namespace alps {

/// \addtogroup alps
/// @{

/// \file parameterlist.h
/// \brief reading a set o parameters
/// 
/// This header contains a class to store a vector of parameters, read from a textual input file, using the old
/// syntax of the 1994 version of these libraries.
///
/// Parameters are specified by single lines containing statements like 
/// \c name \c = \c value
/// where the value needs to be enclosed in double quotes "...." if it contains spaces.
/// The name has to start with a letter, and the next characters can also be numbers or any of the following characters: _'[] .
/// More than one parameter assignment, separated by , or ; can be placed on a single.
///
/// Each set of parameters is enclosed by curly braces {....}. Parameters defined outside of curly braces are global parameters, used for all
/// of the following parameter sets.


/// \brief a vector of Parameters
///
/// each element usually describes the parameters for a single simulation.
/// 
/// The class is derived from a std::vector
class ParameterList : public std::vector<Parameters>
{
public:
  typedef std::vector<Parameters> super_type;
  /// creates an empty vector
  ParameterList() {}
  /// reads Parameters from a std::istream
  ParameterList(std::istream& is) { parse(is); }
  /// reads Parameters from a std::istream
  void parse(std::istream& is);
  /// support for Boost serialization
  template <class ARCHIVE>
  void serialize(ARCHIVE & ar, const unsigned int)
  { ar & static_cast<super_type&>(*this); }

};
/// @}
} // end namespace

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// \brief write the parameters to a std::ostream
///
/// follows the short text-based format and not the XML format
inline std::ostream& operator<<(std::ostream& os,
                                const alps::ParameterList& params)
{
  for (alps::ParameterList::const_iterator it = params.begin();
       it != params.end(); ++it) os << "{\n" << *it << "}\n";
  return os;
}

/// \brief read the parameters from a std::istream
///
/// follows the short text-based format and not the XML format
inline std::istream& operator>>(std::istream& is, alps::ParameterList& params) {
  params.parse(is);
  return is;
}

//
// OSIRIS support
//

#ifndef ALPS_WITHOUT_OSIRIS

/// \brief support for ALPS serialization
inline alps::ODump& operator<<(alps::ODump& od,
                               const alps::ParameterList& p)
{ return od << static_cast<std::vector<alps::Parameters> >(p); }

/// \brief support for ALPS deserialization
inline alps::IDump& operator>>(alps::IDump& id,
                               alps::ParameterList& p)
{ return id >> reinterpret_cast<std::vector<alps::Parameters>&>(p); }

#endif // !ALPS_WITHOUT_OSIRIS

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

//
// XML support
//

#ifndef ALPS_WITHOUT_XML

namespace alps {
/// \addtogroup alps
/// @{

/// \brief Implementation handler of the ALPS XML parser for the ParameterList class  
class ParameterListXMLHandler : public CompositeXMLHandler
{
public:
  ParameterListXMLHandler(ParameterList& list);

protected:  
  void start_child(const std::string& name,
                   const XMLAttributes& attributes,
                   xml::tag_type type);
  void end_child(const std::string& name, xml::tag_type type);

private:
  ParameterList& list_;
  Parameter parameter_;
  Parameters default_, current_;
  ParameterXMLHandler parameter_handler_;
  ParametersXMLHandler current_handler_;
};
/// @}

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

/// \brief XML output of a ParameterList 
///
/// follows the schema on http://xml.comp-phys.org/
inline alps::oxstream& operator<<(alps::oxstream& oxs,
                                  const alps::ParameterList& parameterlist)
{
  oxs << alps::start_tag("PARAMETERLIST");
  alps::ParameterList::const_iterator p_end = parameterlist.end();
  for (alps::ParameterList::const_iterator p = parameterlist.begin();
       p != p_end; ++p) oxs << *p;
  oxs << alps::end_tag("PARAMETERLIST");
  return oxs;
}
#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // !ALPS_WITHOUT_XML

#endif // ALPS_PARSER_PARAMETERLIST_H
