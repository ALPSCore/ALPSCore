/***************************************************************************
* ALPS++ library
*
* xml/xmlparser.h   XML parser
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

#ifndef ALPS_XML_XMLPARSER_H
#define ALPS_XML_XMLPARSER_H

#include <alps/config.h>
#include <alps/parser/xmlhandler.h>

#include <iosfwd>
#include <string>

#if defined(HAVE_XERCES_PARSER)
# include <xercesc/parsers/SAXParser.hpp>
# include <xercesc/sax/HandlerBase.hpp>
#elif defined(HAVE_EXPAT_PARSER)
# include <expat.h>
#endif

namespace alps {

class XMLParser
{
public:
  XMLParser(XMLHandlerBase&);
  ~XMLParser();

  void parse(std::istream& is);
  void parse(const std::string& file);

private:
  XMLParser();

#if defined(HAVE_XERCES_PARSER)
  XERCES_CPP_NAMESPACE_QUALIFIER SAXParser* parser_;
  XERCES_CPP_NAMESPACE_QUALIFIER HandlerBase* handler_;
#elif defined(HAVE_EXPAT_PARSER)
  XML_Parser parser_;
#else
  XMLHandlerBase& handler_;
#endif
};

} // end namespace alps
 
#endif // ALPS_XML_XMLPARSER_H
