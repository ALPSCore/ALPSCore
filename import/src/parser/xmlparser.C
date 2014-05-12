/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/parser/xmlparser.h>
#include <alps/cctype.h>

#include <boost/filesystem/fstream.hpp>
#include <boost/throw_exception.hpp>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#if defined(ALPS_HAVE_XERCES_PARSER)

//
// Xerces C++ XML parser
//

#include <xercesc/parsers/SAXParser.hpp>
#include <xercesc/sax/HandlerBase.hpp>
#include <xercesc/sax/InputSource.hpp>
#include <xercesc/util/BinInputStream.hpp>
#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/util/TransService.hpp>

namespace alps {

namespace detail {

class XMLHandlerAdaptor : public XERCES_CPP_NAMESPACE_QUALIFIER HandlerBase
{
public:
  XMLHandlerAdaptor(XMLHandlerBase& h)
    : XERCES_CPP_NAMESPACE_QUALIFIER HandlerBase(), handler_(h) {}

  void startElement(const XMLCh* const name,
                    XERCES_CPP_NAMESPACE_QUALIFIER AttributeList& attributes) {
    const std::string n =
      XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(name);
    XMLAttributes attr;
    for (std::size_t i = 0; i < attributes.getLength(); ++i) {
      attr.push_back(
        XMLAttribute(
          XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(attributes.getName(i)),
          XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(attributes.getValue(i))));
    }
    handler_.start_element(n, attr, xml::element);
  }
  void endElement(const XMLCh* const name) {
    const std::string n =
      XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(name);
    handler_.end_element(n, xml::element);
  }
  void characters(const XMLCh* const chars, const unsigned int /* length */) {
    std::string t =
      XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(chars);
    std::size_t pi = 0;
    for (std::size_t p = 0; p <= t.length(); ++p) {
      if (t[p] == '\n' || p == t.length()) {
        std::string s = t.substr(pi, p - pi);
        // remove preceding and following blanks
        s = s.erase(0, s.find_first_not_of(' '));
        s = s.erase(s.find_last_not_of(' ')+1);
        if (s.size()) handler_.text(s);
        pi = p + 1;
      }
    }
  }
  void processingInstruction(const XMLCh* const target,
                             const XMLCh* const data) {
    const std::string name =
      XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(target);
    XMLAttributes attr(
      XERCES_CPP_NAMESPACE_QUALIFIER XMLString::transcode(data));
    if (name == "xml-stylesheet") {
      handler_.start_element(name, attr, xml::stylesheet);
      handler_.end_element(name, xml::stylesheet);
    } else {
      handler_.start_element(name, attr, xml::processing_instruction);
      handler_.end_element(name, xml::processing_instruction);
    }
  }

protected:
  XMLHandlerAdaptor();

private:
  XMLHandlerBase& handler_;
};

class IStreamBinInputStream
  : public XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream
{
public:
  IStreamBinInputStream(std::istream& is)
    : XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream(), is_(is) {}
  unsigned int curPos() const { return count_; }
  unsigned int readBytes(XMLByte* const toFill, const unsigned int maxToRead) {
    is_.read(reinterpret_cast<char*>(toFill), maxToRead);
    count_ += is_.gcount();
    return is_.gcount();
  }
private:
  std::istream& is_;
  unsigned int count_;
};

class IStreamBinInputSource
  : public XERCES_CPP_NAMESPACE_QUALIFIER InputSource
{
public:
  IStreamBinInputSource(std::istream& is)
    : XERCES_CPP_NAMESPACE_QUALIFIER InputSource(), stream_(is) {}
  XERCES_CPP_NAMESPACE_QUALIFIER BinInputStream* makeStream() const {
    return new IStreamBinInputStream(stream_);
  }
private:
  std::istream& stream_;
};

} // end namespace detail

XMLParser::XMLParser(XMLHandlerBase& h) {
  XERCES_CPP_NAMESPACE_QUALIFIER XMLPlatformUtils::Initialize();
  parser_ = new XERCES_CPP_NAMESPACE_QUALIFIER SAXParser;
  handler_ = new detail::XMLHandlerAdaptor(h);
  parser_->setDocumentHandler(handler_);
}
XMLParser::~XMLParser() {
  delete parser_;
  XERCES_CPP_NAMESPACE_QUALIFIER XMLPlatformUtils::Terminate();
}

void XMLParser::parse(const std::string& file) {
  parser_->parse(file.c_str());
}
void XMLParser::parse(std::istream& is) {
  detail::IStreamBinInputSource bis(is);
  parser_->parse(bis);
}
void XMLParser::parse(const boost::filesystem::path& file) {
  boost::filesystem::ifstream is(file);
  parse(is);
}

} // namespace alps

#elif defined(ALPS_HAVE_EXPAT_PARSER)

//
// Expat XML Parser
//

namespace alps {

namespace detail {

static void startElement(void* h, const char* name, const char** atts)
{
  XMLHandlerBase* handler = (XMLHandlerBase*)h;
  const std::string n = name;
  XMLAttributes attr;
  for (std::size_t i = 0; atts[i]; i += 2) {
    attr.push_back(XMLAttribute(atts[i], atts[i+1]));
  }
  handler->start_element(n, attr, xml::element);
}
static void endElement(void* h, const char* name)
{
  XMLHandlerBase* handler = (XMLHandlerBase*)h;
  const std::string n = name;
  handler->end_element(n, xml::element);
}
static void characters(void* h, const char* s, int len)
{
  XMLHandlerBase* handler = (XMLHandlerBase*)h;
  std::string t(s, len);
  // remove preceding and following blanks
  t = t.erase(0, t.find_first_not_of(' '));
  t = t.erase(t.find_last_not_of(' ')+1);
  // remove appended CR, NL, \t, etc
  if (t.length() && std::isspace(t[t.length()-1])) t.erase(t.length()-1);
  if (t.length()) handler->text(t);
}
static void processingInstruction(void* h, const char* name, const char* data)
{
  XMLHandlerBase* handler = (XMLHandlerBase*)h;
  const std::string n = name;
  XMLAttributes attr(data);
  if (n == "xml-stylesheet") {
    handler->start_element(n, attr, xml::stylesheet);
    handler->end_element(n, xml::stylesheet);
  } else {
    handler->start_element(n, attr, xml::processing_instruction);
    handler->end_element(n, xml::processing_instruction);
  }
}
static int externalEntity(XML_Parser parser, const char* context,
                          const char* /* base */, const char* systemId,
                          const char* /* publicId */) {
  XML_Parser p = XML_ExternalEntityParserCreate(parser, context, 0);
  std::ifstream fin(systemId);
  char buf[BUFSIZ];
  int done;
  do {
    fin.read(buf, sizeof(buf));
    done = (fin.gcount() < sizeof(buf));
    if (!XML_Parse(p, buf, fin.gcount(), done)) {
      boost::throw_exception(std::runtime_error("XMLParser: " +
        std::string(XML_ErrorString(XML_GetErrorCode(p)))));
    }
  } while (!done);
  XML_ParserFree(p);
  return 1;
}

} // end namespace detail

XMLParser::XMLParser(XMLHandlerBase& h) : parser_(XML_ParserCreate(0)) {
  XML_SetUserData(parser_, &h);
  XML_SetElementHandler(parser_, detail::startElement, detail::endElement);
  XML_SetCharacterDataHandler(parser_, detail::characters);
  XML_SetProcessingInstructionHandler(parser_, detail::processingInstruction);
  XML_SetExternalEntityRefHandler(parser_, detail::externalEntity);
}
XMLParser::~XMLParser() { XML_ParserFree(parser_); }

void XMLParser::parse(std::istream& is) {
  char buf[BUFSIZ];
  do {
    is.read(buf, sizeof(buf));
    if (!XML_Parse(parser_, buf, is.gcount(), (is.gcount() < sizeof(buf))))
      boost::throw_exception(std::runtime_error("XMLParser: " +
        std::string(XML_ErrorString(XML_GetErrorCode(parser_)))));
  } while (is.gcount() == sizeof(buf));
}
void XMLParser::parse(const std::string& file) {
  std::ifstream fin(file.c_str());
  parse(fin);
}
void XMLParser::parse(const boost::filesystem::path& file) {
  boost::filesystem::ifstream fin(file);
  parse(fin);
}

} // namespace alps

#else

//
// native XML Parser
//

#include <alps/parser/parser.h>

namespace alps {

XMLParser::XMLParser(XMLHandlerBase& h) : handler_(h) {}

XMLParser::~XMLParser() {}

void XMLParser::parse(std::istream& in)
{
  while (in) {
    char c;
    in >> c;
    if (!in)
      break;
    in.putback(c);
    if (c=='<') {
      XMLTag tag = parse_tag(in, false);
      if (tag.type == XMLTag::OPENING || tag.type== XMLTag::SINGLE) {
        // start tag
        handler_.start_element(tag.name, tag.attributes, xml::element);
      }
      if (tag.type == XMLTag::CLOSING || tag.type== XMLTag::SINGLE) {
        // end tag
        if (tag.type == XMLTag::CLOSING)
          tag.name.erase(0,1);
        handler_.end_element(tag.name, xml::element);
      }
      if (tag.type == XMLTag::PROCESSING) {
        // processing instruction
        if (tag.name == "xml-stylesheet") {
          handler_.start_element(tag.name, tag.attributes, xml::stylesheet);
          handler_.end_element(tag.name, xml::stylesheet);
        } else {
          handler_.start_element(tag.name, tag.attributes,
                                 xml::processing_instruction);
          handler_.end_element(tag.name, xml::processing_instruction);
        }
      }
    }
    else {
      std::string t = parse_content(in);
      int pi = 0;
      for (std::size_t p = 0; p <= t.length(); ++p) {
        if ( p == t.length() || t[p] == '\n') {
          std::string s = t.substr(pi, p - pi);
          // remove preceding and following blanks
          s = s.erase(0, s.find_first_not_of(' '));
          s = s.erase(s.find_last_not_of(' ') + 1);
          if (s.size()) handler_.text(s);
          pi = p + 1;
        }
      }
    }
  }
}
void XMLParser::parse(const std::string& file) {
  std::ifstream is(file.c_str());
  parse(is);
}
void XMLParser::parse(const boost::filesystem::path& file) {
  boost::filesystem::ifstream is(file);
  parse(is);
}

} // namespace alps

#endif
