/***************************************************************************
* PALM++/xml library
*
* xml/xmlhandler.C   XML handler abstract class
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
  const XMLAttributes& attributes) {
  if (level_ == 0) {
    if (name != basename())
      boost::throw_exception(std::runtime_error("XMLCompositeHandler: unknown start tag : " + name));
    start_top(name, attributes);
  } else if (level_ == 1) {
    if (start_element_impl(name, attributes) == false) {
      map_type::const_iterator h = handlers_.find(name);
      if (h == handlers_.end())
	boost::throw_exception(std::runtime_error("XMLCompositeHandler: unknown start tag : " + name));
      start_child(name, attributes);
      current_ = h->second;
      current_->start_element(name, attributes);
    }
  } else {
    if (current_ == 0) {
      if (start_element_impl(name, attributes) == false) {
	boost::throw_exception(std::runtime_error("XMLCompositeHandler: unknown start tag : " + name));
      }
    } else {
      current_->start_element(name, attributes);
    }
  }
  ++level_;
}
void CompositeXMLHandler::end_element(const std::string& name) {
  if (level_ == 1) {
    end_top(name);
  } else {
    if (current_ == 0) {
      if (end_element_impl(name) == false)
	boost::throw_exception(std::runtime_error("XMLCompositeHandler: unknown end tag : " + name));
    } else {
      current_->end_element(name);
      if (level_ == 2) {
	end_child(name);
	current_ = 0;
      }
    }
  }
  --level_;
}
void CompositeXMLHandler::text(const std::string& text) {
  if (current_ == 0) {
    if (text_impl(text) == false)
      boost::throw_exception(std::runtime_error("XMLCompositeHandler: text is not allowed here"));
  } else {
    current_->text(text);
  }
}
  
} // namespace alps
