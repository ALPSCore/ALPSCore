/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2006 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_XML_XMLHANDLER_H
#define ALPS_XML_XMLHANDLER_H

#include <alps/parser/xmlattributes.h>
#include <alps/parser/xmlstream.h>

#include <boost/throw_exception.hpp>
#include <map>
#include <stdexcept>
#include <string>

namespace alps {

namespace xml {

enum tag_type { element, processing_instruction, stylesheet };

} // namespace xml

class ALPS_DECL XMLHandlerBase
{
public:
  XMLHandlerBase(const std::string& basename) : basename_(basename) {
    if (basename_.empty())
      boost::throw_exception(
        std::invalid_argument("XMLHandlerBase: empty basename"));
  }
  virtual ~XMLHandlerBase() {}

  void set_basename(const std::string&) {
    if (basename_.empty())
      boost::throw_exception(
        std::invalid_argument("XMLHandlerBase: empty basename"));
  }
  std::string basename() const { return basename_; }

  virtual void start_element(const std::string& name,
                             const XMLAttributes& attributes,
                             xml::tag_type type) = 0;
  virtual void end_element(const std::string& name, xml::tag_type type) = 0;
  virtual void text(const std::string& text) = 0;

private:
  XMLHandlerBase();
  std::string basename_;
};


class ALPS_DECL DummyXMLHandler : public XMLHandlerBase
{
public:
  DummyXMLHandler(const std::string& basename)
    : XMLHandlerBase(basename) {}
  void start_element(const std::string& /* name */,
                     const XMLAttributes& /* attributes */,
                     xml::tag_type /* type */) {}
  void end_element(const std::string& /* name */, xml::tag_type /* type */) {}
  void text(const std::string& /* text */) {}
};


template<class T>
class SimpleXMLHandler : public XMLHandlerBase
{
public:
  typedef T value_type;

  SimpleXMLHandler(const std::string& basename, T& val,
                   const std::string& attr = "")
    : XMLHandlerBase(basename), value_(val), attr_(attr), started_(false) {}
  virtual ~SimpleXMLHandler() {}

  virtual void start_element(const std::string& name,
                             const XMLAttributes& attributes,
                             xml::tag_type type) {
    if (type == xml::element) {
      if (name != basename())
        boost::throw_exception(std::runtime_error(
          "SimpleXMLHandler::start_element: unknown start tag <" + name +
          ">"));
      if (started_)
        boost::throw_exception(std::runtime_error(
          "SimpleXMLHandler::start_element: encountered nested start tags <" +
          name + ">"));
      if (!attr_.empty()) {
        if (!attributes.defined(attr_))
          boost::throw_exception(std::runtime_error(
            "SimpleXMLHandler::start_element: attribute \"" + attr_ +
            "\" not defined in <" + name + "> tag"));
        value_ = boost::lexical_cast<value_type>(attributes[attr_]);
      }
      started_ = true;
    }
  }

  virtual void end_element(const std::string& name, xml::tag_type type) {
    if (type == xml::element) {
      if (name != "" && name != basename())
        boost::throw_exception(std::runtime_error(
          "SimpleXMLHandler::end_element: unknown end tag </" + name + ">"));
      if (!started_)
        boost::throw_exception(std::runtime_error(
          "SimpleXMLHandler::end_element: unbalanced end tag </" + basename() +
          ">"));
      if (attr_.empty()) {
        value_ = boost::lexical_cast<value_type>(buffer_);
        buffer_.clear();
      }
      started_ = false;
    }
  }

  virtual void text(const std::string& text) {
    if (attr_.empty()) {
      if (!buffer_.empty()) buffer_ += ' ';
      buffer_ += text;
    }
  }

private:
  value_type& value_;
  std::string attr_;
  bool started_;
  std::string buffer_;
};


class ALPS_DECL CompositeXMLHandler : public XMLHandlerBase
{
private:
  typedef XMLHandlerBase base_type;
  typedef std::map<std::string, XMLHandlerBase*> map_type;

public:
  CompositeXMLHandler(const std::string& basename)
    : base_type(basename), handlers_(), current_(0), level_(0) {}
  virtual ~CompositeXMLHandler() {}

  void clear_handler() { handlers_.clear(); }
  void add_handler(XMLHandlerBase& handler);
  bool has_handler(const XMLHandlerBase& handler) const;
  bool has_handler(const std::string& name) const;

  void start_element(const std::string& name,
                     const XMLAttributes& attributes,
                     xml::tag_type type);
  void end_element(const std::string& name, xml::tag_type type);
  void text(const std::string& text);

protected:
  virtual void start_top(const std::string& /* name */,
                         const XMLAttributes& /* attributes */,
                         xml::tag_type /* type */) {}
  virtual void end_top(const std::string& /* name */,
                       xml::tag_type /* type */) {}
  virtual void start_child(const std::string& /* name */,
                           const XMLAttributes& /* attributes */,
                           xml::tag_type /* type */) {}
  virtual void end_child(const std::string& /* name */,
                         xml::tag_type /* type */) {}

  virtual bool start_element_impl(const std::string& /* name */,
                                  const XMLAttributes& /* attributes */,
                                  xml::tag_type /* type */)
  { return false; }
  virtual bool end_element_impl(const std::string& /* name */,
                                xml::tag_type /* type */)
  { return false; }
  virtual bool text_impl(const std::string& /* text */) { return false; }

private:
  map_type handlers_;       // list of pointer to handlers
  XMLHandlerBase* current_; // pointer to current handler
  unsigned int level_;
};


template<class T, class C = std::vector<T>, class H = SimpleXMLHandler<T> >
class VectorXMLHandler : public CompositeXMLHandler
{
public:
  VectorXMLHandler(const std::string& basename, C& cont, T& val, H& handler)
    : CompositeXMLHandler(basename), cont_(cont), val_(val),
      handler_(handler) {
    CompositeXMLHandler::add_handler(handler_);
  }
  virtual ~VectorXMLHandler() {}

protected:
  virtual void end_child(const std::string& name, xml::tag_type type) {
    if (type == xml::element && name == handler_.basename())
      cont_.push_back(val_);
  }

private:
  C& cont_;
  T& val_;
  H& handler_;
};


class ALPS_DECL PrintXMLHandler : public XMLHandlerBase
{
public:
  PrintXMLHandler(std::ostream& os = std::cout)
    : XMLHandlerBase("printer"), oxs_(os), in_text_(false) {}

  void start_element(const std::string& name,
                     const XMLAttributes& attributes,
                     xml::tag_type type) {
    in_text_ = false;
    switch (type) {
    case xml::element :
      oxs_ << start_tag(name)
           << attributes;
      break;
    case xml::processing_instruction :
      oxs_ << processing_instruction(name)
           << attributes;
    case xml::stylesheet :
      oxs_ << stylesheet(attributes["href"]);
    default :
      break;
    }
  }
  void end_element(const std::string& name, xml::tag_type type) {
    in_text_ = false;
    switch (type) {
    case xml::element :
      oxs_ << end_tag(name);
      break;
    case xml::processing_instruction :
    case xml::stylesheet :
      // nothing to do
    default :
      break;
    }
  }
  void text(const std::string& text) {
    if (in_text_) oxs_ << '\n';
    oxs_ << text;
    in_text_ = true;
  }

private:
  oxstream oxs_;
  bool in_text_;
};


class ALPS_DECL StylesheetXMLHandler : public XMLHandlerBase
{
public:
  StylesheetXMLHandler(std::string& style)
    : XMLHandlerBase("style"), style_(style) {}
  void start_element(const std::string&,
                     const XMLAttributes& attributes,
                     xml::tag_type type) {
    if (type == xml::stylesheet) style_ = attributes["href"];
  }
  void end_element(const std::string&, xml::tag_type) {}
  void text(const std::string&) {}
private:
  std::string& style_;
};

} // namespace alps

#endif // ALPS_XML_XMLHANDLER_H
