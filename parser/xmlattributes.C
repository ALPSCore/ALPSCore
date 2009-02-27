/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/parser/xmlattributes.h>
#include <boost/classic_spirit.hpp>

namespace alps {

namespace {

struct assign_string {
  assign_string(const std::string& name)
    : ptr_(const_cast<std::string*>(&name)) {}
  void operator()(const char* first, const char* last) const
  { *ptr_ = std::string(first, last); }
  std::string* ptr_;
};

struct append_attr {
  append_attr(const XMLAttributes& attr, const std::string& name,
              const std::string& value)
    : ptr_(const_cast<XMLAttributes*>(&attr)), name_(name), value_(value) {}
  void operator()(const char*, const char*) const
  { ptr_->push_back(XMLAttribute(name_, value_)); }
  XMLAttributes* ptr_;
  const std::string& name_;
  const std::string& value_;
};

struct attr_parser : public boost::spirit::grammar<attr_parser> {
  attr_parser(XMLAttributes& a) : attr(a) {}

  template<typename ScannerT>
  struct definition {
    boost::spirit::rule<ScannerT> name_p, value_p, attribute_p, attributes_p;
    definition(const attr_parser& self)
    {
      using boost::spirit::alpha_p;
      using boost::spirit::alnum_p;
      using boost::spirit::anychar_p;
      using boost::spirit::ch_p;
      name_p = ((alpha_p | ch_p('_')) >> *(alnum_p | ch_p('_')))[assign_string(self.name)];
      value_p = ch_p('\"') >> (*(anychar_p - ch_p('\"')))[assign_string(self.value)] >> ch_p('\"');
      attributes_p = *(name_p >> ch_p('=') >> value_p)[append_attr(self.attr, self.name, self.value)];
    }
    const boost::spirit::rule<ScannerT>& start() const { return attributes_p; }
  };

  XMLAttributes& attr;
  std::string name, value;
};

}

XMLAttributes::XMLAttributes(const std::string& str)
{
  if (!boost::spirit::parse(str.c_str(), attr_parser(*this),
    boost::spirit::space_p).full)
      boost::throw_exception(std::runtime_error("parse failed"));
}

void XMLAttributes::push_back(const XMLAttribute& attr)
{
  if (defined(attr.key()))
    boost::throw_exception(std::runtime_error("duplicated attribute " +
      attr.key()));
  map_[attr.key()] = list_.size();
  list_.push_back(attr);
}

} // namespace alps
