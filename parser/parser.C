/***************************************************************************
* PALM++/xml library
*
* xml/xml.C     a simple XML parser
*
* $Id$
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Prakash Dayal <prakash@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
n* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#include <alps/parser/parser.h>

#include <alps/cctype.h>

#include <boost/throw_exception.hpp>
#include <iostream>
#include <stdexcept>
#include <string>

namespace alps {

std::string parse_identifier(std::istream& in) 
{
  char c;
  in >> c;
  std::string name;
  while (detail::is_identifier_char(c)) {
    name+=c;
    c=in.get();
  }
  in.putback(c);
  return name;
}

std::string read_until(std::istream& in, char end) 
{
  std::string s;
  char c;
  in >> c;
  while(c!=end && in) {
    s+=c;
    c=in.get();
  }
  if (c!=end)
    boost::throw_exception(std::runtime_error(std::string("read past end of stream while scanning for ")+end));
  s = s.erase(s.find_last_not_of(" \t\r\n")+1);
  return s;
}

void check_character(std::istream& in, char test, const std::string& error) 
{
  char c;
  in >> c;
  if (c!=test)
    boost::throw_exception(std::runtime_error(error));
}

// private functions
namespace detail {

std::string parse_string(std::istream& in)
{
  check_character(in,'"',"string expected as attribute value");
  return read_until(in,'"');
}

std::string xml_parse_name(std::istream& in)
{
 std::string the_string;
 char c;
 in>>c;
 the_string=c;
 if(c=='!' || c=='?')
   return the_string;

  // copy following alphanumeric characters or /,:,_ into the string
  c=in.get();
  while ((std::isalnum(c) || (c=='/') || (c==':') || (c=='_')) &&in) {
    the_string += c;
    c=in.get();
  }
  in.putback(c);
  return the_string;
}

void xml_read_attribute(std::istream& in, std::string& name, std::string& value)
{
  name=xml_parse_name(in);
  if(name=="")
    boost::throw_exception(std::runtime_error("attribute expected"));
  check_character(in,'=',"= expected after attribute name " + name);
  value=parse_string(in);
}

std::string xml_read_tag(std::istream& in)
{
  check_character(in,'<',"XML tag expected");
  return xml_parse_name(in);
}

void xml_close_tag(std::istream& in)
{
  check_character(in,'>',"closing > of tag expected");
}

void xml_close_single_tag(std::istream& in)
{
  check_character(in,'/',"closing /> of tag expected");
  check_character(in,'>',"closing /> of tag expected");
}

void skip_comment(std::istream& in)
{
  char c;
  do {
    in >> c;
    if(c=='"') read_until(in,'"');
  } while (c!='>'&&in);
}

} // namespace detail

XMLTag parse_tag(std::istream& in, bool skip_comments)
{
  XMLTag tag;
  tag.name = detail::xml_read_tag(in);
  if(tag.name=="?") {
    tag.type=XMLTag::COMMENT;
    detail::skip_comment(in);
  }
  else if( tag.name=="!") {
    tag.type=XMLTag::PROCESSING;
    detail::skip_comment(in);
  }
  else {
    if(tag.name[0]=='/')
      tag.type=XMLTag::CLOSING;
    else if (tag.name[tag.name.size()-1]=='/') {
      tag.name.erase(tag.name.size()-1,1);
      tag.type=XMLTag::SINGLE;
    }
    else
      tag.type=XMLTag::OPENING;
    if (tag.type!=XMLTag::OPENING) {
      detail::xml_close_tag(in);
      return tag;
    }
    

    std::string n,v;
    char c;
    in>>c;
    while ((c!='/')&&(c!='>')) {
      in.putback(c);
      detail::xml_read_attribute(in,n,v);
      tag.attributes[n]=v;
      in>>c;
    }
    
    if(c=='/') {
      tag.type=XMLTag::SINGLE;
      in>>c;
    }
    in.putback(c);
    detail::xml_close_tag(in);
  } // matches outer most else
  return ((skip_comments && !tag.is_element()) ? parse_tag(in,true) : tag);
}

// Parse some contents
std::string parse_content(std::istream& in)
{
  std::string val=read_until(in,'<'); 
  in.putback('<');
  return val;
}


void skip_element(std::istream& in, const XMLTag& start)
{
  if (start.type != XMLTag::OPENING)
    return;
  while(true) {
    parse_content(in); 
    XMLTag t = parse_tag(in);
    if (t.is_element()) {
      if (t.type==XMLTag::CLOSING) {
	if (t.name == ("/" + start.name))
	  break;
	else {
	  boost::throw_exception(std::runtime_error("illegal closing tag in XML"));}
      }
      else
	skip_element(in,t);
    }
  } // end of while(true)
}

void check_tag(std::istream& in, const std::string& name) 
{
  XMLTag tag=parse_tag(in);
  if (tag.name!=name)
    boost::throw_exception(std::runtime_error("Encountered tag <" + tag.name + "> instead of <" + name + ">" ));
}

} // namespace alps
