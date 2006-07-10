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

#include <alps/parser/xmlhandler.h>

namespace alps {

void CompositeXMLHandler::add_handler(XMLHandlerBase& handler) {
  if (handlers_.find(handler.basename()) != handlers_.end())
    boost::throw_exception(std::invalid_argument("XMLHandlerSet: duplicated handler for tag : " + handler.basename()));
  handlers_[handler.basename()] = &handler;
}
bool CompositeXMLHandler::has_handler(const XMLHandlerBase& handler) const {
  return handlers_.find(handler.basename()) != handlers_.end();
}
bool CompositeXMLHandler::has_handler(const std::string& name) const {
  return handlers_.find(name) != handlers_.end();
}
  
void CompositeXMLHandler::start_element(const std::string& name,
  const XMLAttributes& attributes, xml::tag_type type) {
  if (level_ == 0) {
    if (type == xml::element && name != basename())
      boost::throw_exception(std::runtime_error(
        "XMLCompositeHandler: unknown start tag : " + name));
    start_top(name, attributes, type);
  } else if (level_ == 1) {
    if (start_element_impl(name, attributes, type) == false) {
      map_type::const_iterator h = handlers_.find(name);
      if (h == handlers_.end())
        boost::throw_exception(std::runtime_error(
          "XMLCompositeHandler: unknown start tag : " + name));
      start_child(name, attributes, type);
      current_ = h->second;
      current_->start_element(name, attributes, type);
    }
  } else {
    if (current_ == 0) {
      if (start_element_impl(name, attributes, type) == false) {
        boost::throw_exception(std::runtime_error(
          "XMLCompositeHandler: unknown start tag : " + name));
      }
    } else {
      current_->start_element(name, attributes, type);
    }
  }
  ++level_;
}
void CompositeXMLHandler::end_element(const std::string& name,
                                      xml::tag_type type) {
  if (level_ == 1) {
    end_top(name, type);
  } else {
    if (current_ == 0) {
      if (end_element_impl(name, type) == false)
        boost::throw_exception(std::runtime_error(
          "XMLCompositeHandler: unknown end tag : " + name));
    } else {
      current_->end_element(name, type);
      if (level_ == 2) {
        end_child(name, type);
        current_ = 0;
      }
    }
  }
  --level_;
}
void CompositeXMLHandler::text(const std::string& text) {
  if (current_ == 0) {
    if (text_impl(text) == false)
      boost::throw_exception(std::runtime_error(
        "XMLCompositeHandler: text is not allowed here"));
  } else {
    current_->text(text);
  }
}
  
} // namespace alps
