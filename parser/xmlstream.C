/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2005 by Synge Todo <wistaria@comp-phys.org>
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

#include <alps/parser/xmlstream.h>

// some file (probably a python header) defines a tolower macro ...
#undef tolower
#undef toupper

#include <boost/regex.hpp>
#include <boost/throw_exception.hpp>
#include <stdexcept>
#include <sstream>
#include <string>

namespace alps {

oxstream::oxstream()
  : of_(), os_(std::cout), stack_(), attr_(), context_(NotSpecified),
    linebreak_(true), offset_(0), offset_incr_(2) {}
oxstream::oxstream(std::ostream& os, uint32_t incr)
  : of_(), os_(os), stack_(), attr_(), context_(NotSpecified),
    linebreak_(true), offset_(0), offset_incr_(incr) {}
oxstream::oxstream(const boost::filesystem::path& file, uint32_t incr)
  : of_(file), os_(of_), stack_(), attr_(), context_(NotSpecified),
    linebreak_(true), offset_(0), offset_incr_(incr) {}
oxstream::~oxstream() {
  output();
  if (stack_.size() != 0) {
    std::cerr << "WARNING: Unclosed tag: " << stack_.top().first << "!\n";
    // ATTN: destructor should not throw
    //boost::throw_exception(std::runtime_error(
    //  "unclosed tag: " + stack_.top().first));
    }
}

oxstream& oxstream::operator<<(const detail::header_t& c)
{
  if (context_ == Comment || context_ == Cdata)
    boost::throw_exception(std::runtime_error(
      "header not allowed in comment or CDATA section"));
    
  *this << detail::pi_t("xml")
        << detail::attribute_t("version", c.version);
  if (!c.encoding.empty())
    *this << detail::attribute_t("encoding", c.encoding);
  return *this;
}

oxstream& oxstream::operator<<(const detail::start_tag_t& c)
{
  if (context_ == Comment || context_ == Cdata)
    boost::throw_exception(std::runtime_error(
      "start tag not allowed in comment or CDATA section"));

  output();
  stack_.push(std::make_pair(c.name, linebreak_));
  attr_.clear();
  context_ = StartTag;
  return *this;
}

oxstream& oxstream::operator<<(const detail::end_tag_t& c)
{
  if (!c.name.empty() && c.name != stack_.top().first)
    boost::throw_exception(std::runtime_error("inconsistent end tag name: " +  c.name + " does not agree with " + stack_.top().first));
  output(true);
  return *this;
}

oxstream& oxstream::operator<<(const XMLAttribute& c)
{
  if (context_ != StartTag && context_ != PI)
    boost::throw_exception(std::runtime_error(
      "attribute is allowed only in tag"));
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
        offset_ += offset_incr_;
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
      offset_ -= offset_incr_;
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
  boost::regex_replace(oi, str.begin(), str.end(), 
                       boost::regex("(&)|(')|(>)|(<)|(\")"), 
                       "(?1&amp;)(?2&apos;)(?3&gt;)(?4&lt;)(?5&quot;)",
                       boost::match_default | boost::format_all);
  return out.str();
}

oxstream& oxstream::operator<<(const detail::stylesheet_t& c)
{
  (*this) << processing_instruction("xml-stylesheet")
          << attribute("type","text/xsl") << attribute("href",c.url);
  return *this;
}

} // end namespace alps

// workaround for boost 1_32_0
#if !defined(BOOST_REGEX_WORKAROUND_HPP)
# if defined(__GNUC__) && !defined(__ICC) && !defined(__ECC) && __GNUC__ == 3 && __GNUC_MINOR__ < 3 // ICC 8.0 defines __GNUC__!
template boost::re_detail::perl_matcher<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > >, std::allocator<boost::sub_match<__gnu_cxx::__normal_iterator<char const*, std::basic_string<char, std::char_traits<char>, std::allocator<char> > > > >, boost::regex_traits<char>, std::allocator<char> >;
# endif
#endif
