/***************************************************************************
* ALPS++ library
*
* parser/attributes.h
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

#ifndef ALPS_PARSER_ATTRIBUTES_H
#define ALPS_PARSER_ATTRIBUTES_H

#include <boost/lexical_cast.hpp>
#include <boost/throw_exception.hpp>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace alps {

class XMLAttribute
{
public:
  typedef std::string key_type;
  typedef std::string value_type;

  XMLAttribute(const XMLAttribute& attr)
    : key_(attr.key_), value_(attr.value_) {}
  XMLAttribute(const key_type& k) : key_(k), value_() {}
  XMLAttribute(const key_type& k, const value_type& v) : key_(k), value_(v) {}
  XMLAttribute(const key_type& k, const char * v) : key_(k), value_(v) {}
  template<class T>
  XMLAttribute(const key_type& k, const T& v)
    : key_(k), value_(boost::lexical_cast<std::string, T>(v)) {}

  key_type& key() { return key_; }
  const key_type& key() const { return key_; }
  value_type& value() { return value_; }
  const value_type& value() const { return value_; }

private:
  key_type key_;
  value_type value_;
};

class XMLAttributes
{
public:
  typedef XMLAttribute::key_type    key_type;
  typedef XMLAttribute::value_type  value_type;

  typedef std::vector<XMLAttribute> list_type;
  typedef list_type::size_type      size_type;

  typedef std::map<key_type, size_type> map_type;

  typedef XMLAttribute *            pointer_type;
  typedef const XMLAttribute *      const_pointer_type;
  typedef XMLAttribute &            reference_type;
  typedef const XMLAttribute &      const_reference_type;
  typedef list_type::iterator       iterator;
  typedef list_type::const_iterator const_iterator;

  XMLAttributes() {}

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
      boost::throw_exception(std::runtime_error("attribute not defined"));
    return list_[map_.find(k)->second].value();
  }
  value_type value_or_default(const key_type& k, const value_type& v) const {
    return defined(k) ? (*this)[k] : v;
  }

  iterator begin() { return list_.begin(); }
  const_iterator begin() const { return list_.begin(); }
  iterator end() { return list_.end(); }
  const_iterator end() const { return list_.end(); }

  void push_back(const XMLAttribute& attr) {
    if (defined(attr.key()))
      boost::throw_exception(std::runtime_error("duplicated attribute " + attr.key()));
    map_[attr.key()] = list_.size();
    list_.push_back(attr);
  }
  void push_back(const key_type& k, const value_type& v) {
    push_back(XMLAttribute(k, v));
  }
  XMLAttributes& operator<<(const XMLAttribute& a) {
    (*this)[a.key()] = a.value();
    return *this;
  }
  XMLAttributes& operator<<(const XMLAttributes& attr) {
    for (const_iterator itr = attr.begin(); itr != attr.end(); ++itr)
      (*this) << *itr;
    return *this;
  }

private:
  list_type list_;
  map_type map_;
};

} // namespace alps

#endif // ALPS_PARSER_ATTRIBUTE_H
