/***************************************************************************
* ALPS/parser library
*
* alps/parser/xmlstream.C   XML stream class
*
* $Id$
*
* Copyright (C) 2001-2003 by Synge Todo <wistaria@comp-phys.org>,
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

#include <alps/parser/xmlstream.h>

#include <boost/regex.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <sstream>
#include <string>

namespace alps {

oxstream::oxstream(std::ostream& os, uint32_t incl)
  : of_(), os_(os), stack_(), attr_(), context_(NotSpecified),
    linebreak_(true), offset_(0), offset_incl_(incl) {}
oxstream::oxstream(const boost::filesystem::path& file, uint32_t incl)
  : of_(file), os_(of_), stack_(), attr_(), context_(NotSpecified),
    linebreak_(true), offset_(0), offset_incl_(incl) {}
oxstream::~oxstream() {
  output();
  if (stack_.size() != 0)
    boost::throw_exception(std::runtime_error("unclosed tags exist"));
}

oxstream& oxstream::operator<<(const detail::header_t& c)
{
  if (context_ == Comment || context_ == Cdata)
    boost::throw_exception(std::runtime_error("header not allowed in comment or CDATA section"));
    
  *this << detail::pi_t("xml")
	<< detail::attribute_t("version", c.version);
  if (!c.encoding.empty())
    *this << detail::attribute_t("encoding", c.encoding);
  return *this;
}

oxstream& oxstream::operator<<(const detail::start_tag_t& c)
{
  if (context_ == Comment || context_ == Cdata)
    boost::throw_exception(std::runtime_error("start tag not allowed in comment or CDATA section"));

  output();
  stack_.push(std::make_pair(c.name, linebreak_));
  attr_.clear();
  context_ = StartTag;
  return *this;
}

oxstream& oxstream::operator<<(const detail::end_tag_t& c)
{
  if (!c.name.empty() && c.name != stack_.top().first)
    boost::throw_exception(std::runtime_error("inconsistent end tag name"));
  output(true);
  return *this;
}

oxstream& oxstream::operator<<(const XMLAttribute& c)
{
  if (context_ != StartTag && context_ != PI)
    boost::throw_exception(std::runtime_error("attribute is allowed only in tag"));
  attr_.push_back(c);
  return *this;
}

oxstream& oxstream::operator<<(const XMLAttributes& c)
{
  for (XMLAttributes::const_iterator itr = c.begin(); itr != c.end(); ++itr)
    *this << *itr;
  return *this;
}

oxstream& oxstream::operator<<(const detail::attribute_t& c)
{
  return (*this << c.attr);
}

oxstream& oxstream::operator<<(const detail::pi_t& c)
{
  *this << static_cast<detail::start_tag_t>(c);
  context_ = PI;
  return *this;
}

oxstream& oxstream::start_comment()
{
  output();
  if (linebreak_) for (uint32_t i = 0; i < offset_; ++i) os_ << ' ';
  os_ << "<!-- ";
  context_ = Comment;
  return *this;
}

oxstream& oxstream::end_comment()
{
  if (context_ != Comment)
    boost::throw_exception(std::runtime_error("not in comment context"));
  os_ << " -->";
  if (linebreak_) os_ << '\n';
  context_ = NotSpecified;
  return *this;
}

oxstream& oxstream::no_linebreak()
{
  if (context_ != StartTag)
    boost::throw_exception(std::runtime_error("no_linebreak is allowed only in starttag"));
  linebreak_ = false;
  return *this;
}

oxstream& oxstream::text_str(const std::string& text)
{
  if (context_ == Comment || context_ == Cdata) {
    // just output
    os_ << text;
  } else {
    if (context_ != Text) {
      output();
      if (linebreak_) for (uint32_t i = 0; i < offset_; ++i) os_ << ' ';
      context_ = Text;
    }
    std::size_t pos0 = 0;
    while (true) {
      std::size_t pos1 = text.find('\n', pos0);
      os_ << text.substr(pos0, pos1);
      if (pos1 == std::string::npos) break;
      if (linebreak_) {
	os_ << '\n';
	for (uint32_t i = 0; i < offset_; ++i) os_ << ' ';
      } else {
	os_ << ' ';
      }
      pos0 = pos1 + 1;
    }
  }
  return *this;
}

void oxstream::output(bool close)
{
  if (context_ == StartTag || context_ == PI) {
    output_offset();
    if (context_ == PI) {
      os_ << "<?" << stack_.top().first;
    } else {
      os_ << "<" << stack_.top().first;
    }

    for (XMLAttributes::const_iterator a = attr_.begin();
	 a != attr_.end(); ++a)
      os_ << " " << a->key() << "=\"" << static_cast<std::string>(a->value())
	  << "\"";
    
    if (context_ == PI) {
      os_ << "?>";
    } else {
      if (close) {
	os_ << "/>";
      } else {
	os_ << ">";
	offset_ += offset_incl_;
      }
    }
    if (context_ == PI || close) {
      linebreak_ = stack_.top().second;
      stack_.pop();
    }
    if (linebreak_) os_ << std::endl;
    context_ = NotSpecified;
  } else {
    if (context_ == Text) {
      if (linebreak_) os_ << std::endl;
      context_ = NotSpecified;
    }
    if (close) {
      offset_ -= offset_incl_;
      if (linebreak_) for (uint32_t i = 0; i < offset_; ++i) os_ << ' ';
      os_ << "</" << stack_.top().first << ">";
      linebreak_ = stack_.top().second;
      stack_.pop();
      if (linebreak_) os_ << std::endl;
    }
  }
}

void oxstream::output_offset()
{
  if (stack_.size() != 0  && stack_.top().second)
    for (uint32_t i = 0; i < offset_; ++i) os_ << ' ';
}

std::string convert(const std::string& str)
{
  std::ostringstream out;
  std::ostream_iterator<char> oi(out);
  boost::regex_merge(oi, str.begin(), str.end(), 
		     boost::regex("(&)|(')|(>)|(<)|(\")"), 
		     "(?1&amp;)(?2&apos;)(?3&gt;)(?4&lt;)(?5&quot;)",
		     boost::match_default | boost::format_all);
  return out.str();
}

} // end namespace alps


#ifdef __GNUC__
template boost::re_detail::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char>, std::allocator<char> >;
#endif
