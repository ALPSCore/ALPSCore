/***************************************************************************
* ALPS++ library
*
* parser/parameterlist.h   An array of parameters
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_PARSER_PARAMETERLIST_H
#define ALPS_PARSER_PARAMETERLIST_H

#include <alps/config.h>
#include <alps/parameters.h>

#ifndef ALPS_WITHOUT_OSIRIS
# include <alps/osiris.h>
#endif
#ifndef ALPS_WITHOUT_XML
# include <alps/xml.h>
#endif

#include <iostream>
#include <vector>

namespace alps {

class ParameterList : public std::vector<Parameters>
{
public:
  ParameterList() {}
  ParameterList(std::istream& is) { parse(is); }
  void parse(std::istream& is);
};

} // end namespace

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline std::ostream& operator<<(std::ostream& os,
				const alps::ParameterList& params)
{
  for (alps::ParameterList::const_iterator it = params.begin();
       it != params.end(); ++it) os << "{\n" << *it << "}\n";
  return os;
}

inline
std::istream& operator>>(std::istream& is, alps::ParameterList& params) {
  params.parse(is);
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

inline alps::ODump& operator<<(alps::ODump& od,
			       const alps::ParameterList& p)
{ return od << static_cast<std::vector<alps::Parameters> >(p); }

inline alps::IDump& operator>>(alps::IDump& id,
			       alps::ParameterList& p)
{ return id >> reinterpret_cast<std::vector<alps::Parameters>&>(p); }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // !ALPS_WITHOUT_OSIRIS


//
// XML support
//

#ifndef ALPS_WITHOUT_XML

namespace alps {

class ParameterListXMLHandler : public CompositeXMLHandler
{
public:
  ParameterListXMLHandler(ParameterList& list);

protected:  
  void start_child(const std::string& name,
		   const XMLAttributes& attributes);
  void end_child(const std::string& name);

private:
  ParameterList& list_;
  Parameter parameter_;
  Parameters default_, current_;
  ParameterXMLHandler parameter_handler_;
  ParametersXMLHandler current_handler_;
};

} // namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

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
