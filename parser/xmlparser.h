/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_XML_XMLPARSER_H
#define ALPS_XML_XMLPARSER_H

#include <alps/config.h>
#include <alps/parser/xmlhandler.h>

#include <boost/filesystem/path.hpp>
#include <iosfwd>
#include <string>

#if defined(ALPS_HAVE_XERCES_PARSER)
# include <xercesc/parsers/SAXParser.hpp>
# include <xercesc/sax/HandlerBase.hpp>
#elif defined(ALPS_HAVE_EXPAT_PARSER)
# include <expat.h>
#endif

namespace alps {

class ALPS_DECL XMLParser
{
public:
  XMLParser(XMLHandlerBase&);
  ~XMLParser();

  void parse(std::istream& is);
  void parse(const std::string& file);
  void parse(const boost::filesystem::path& file);

private:
  XMLParser();

#if defined(ALPS_HAVE_XERCES_PARSER)
  XERCES_CPP_NAMESPACE_QUALIFIER SAXParser* parser_;
  XERCES_CPP_NAMESPACE_QUALIFIER HandlerBase* handler_;
#elif defined(ALPS_HAVE_EXPAT_PARSER)
  XML_Parser parser_;
#else
  XMLHandlerBase& handler_;
#endif
};

} // end namespace alps

#endif // ALPS_XML_XMLPARSER_H
