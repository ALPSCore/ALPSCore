/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Prakash Dayal <prakash@comp-phys.org>
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

std::string parse_parameter_name(std::istream& in) 
{
  char c;
  in >> c;
  std::string name;
  while (in && !in.eof() && (detail::is_identifier_char(c) || c=='\''  || c=='[')) {
    name+=c;
        if (c=='[') 
          do {
            c=in.get();
                name+=c;
          } while (c!=']');
    c=in.get();
  }
  if (!in.eof())
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

ALPS_DECL void check_character(std::istream& in, char test, const std::string& error) 
{
  char c;
  in >> c;
  if (c!=test)
    boost::throw_exception(std::runtime_error(error));
}

// private functions
namespace detail {

/// reads a string, enclosed in double quotes "
std::string parse_string(std::istream& in)
{
  check_character(in,'"',"string expected as attribute value");
  return read_until(in,'"');
}

/// reads any legal XML tag or atrribute name
std::string xml_parse_name(std::istream& in)
{
 std::string the_string;
 char c;
 in>>c;
 the_string=c;
 if(c=='!' || c=='?')
   return the_string;

  // copy following alphanumeric characters or /,:,_,-,. into the string
  c=in.get();
  while ((std::isalnum(c) || (c=='/') || (c==':') || (c=='_') || (c=='-') ||
          (c=='.')) && in) {
    the_string += c;
    c=in.get();
  }
  in.putback(c);
  return the_string;
}

/// parses an XML attribute 
void xml_read_attribute(std::istream& in, std::string& name, std::string& value)
{
  name=xml_parse_name(in);
  if(name=="")
    boost::throw_exception(std::runtime_error("attribute expected"));
  check_character(in,'=',"= expected after attribute name " + name);
  value=parse_string(in);
}

/// parses the opening < and the name of a tag
std::string xml_read_tag(std::istream& in)
{
  check_character(in,'<',"XML tag expected");
  return xml_parse_name(in);
}

/// checks for a closing > of a tag
void xml_close_tag(std::istream& in)
{
  check_character(in,'>',"closing > of tag expected");
}

/// checks for a closing /> of a tag
void xml_close_single_tag(std::istream& in)
{
  check_character(in,'/',"closing /> of tag expected");
  check_character(in,'>',"closing /> of tag expected");
}

/// skips over an XML comment or processing instruction
void skip_comment(std::istream& in, bool processing=false)
{
  char c;
  int dashcount =0;
  do {
    in >> c;
    if(!processing && c=='-')
      ++dashcount;
    else if(processing && c=='?')
      dashcount=2;
    else if (c != '>')
      dashcount=0;
    if(c=='"') read_until(in,'"');
  } while ((dashcount<2 || c!='>')&&in);
}

} // namespace detail

XMLTag parse_tag(std::istream& in, bool skip_comments)
{
  XMLTag tag;
  tag.name = detail::xml_read_tag(in);
  if(tag.name=="?") {
    tag.type=XMLTag::PROCESSING;
    tag.name = detail::xml_parse_name(in);
    std::string n,v;
    char c;
    in >> c;
    while (c!='?') {
      in.putback(c);
      detail::xml_read_attribute(in, n, v);
      tag.attributes[n]=v;
      in >> c;
    }
    detail::skip_comment(in,true);
  }
  else if( tag.name=="!") {
    tag.type=XMLTag::COMMENT;
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
