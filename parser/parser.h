/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2004 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_PARSER_PARSER_H
#define ALPS_PARSER_PARSER_H

#include <alps/cctype.h>
#include <map>
#include <string>

namespace alps {

namespace detail {

inline bool is_identifier_char(char c) 
{ 
  return std::isalnum(c) || c=='_' || c=='\'' || c==':';
}

} // end namespace detail

extern std::string parse_identifier(std::istream& in);

extern std::string parse_parameter_name(std::istream& in);

extern std::string read_until(std::istream& in, char end);

extern void check_character(std::istream& in, char c, const std::string& err);

struct XMLTag
{
  XMLTag() {}
  typedef std::map<std::string,std::string> AttributeMap;
  std::string name;
  AttributeMap attributes;
  enum {OPENING, CLOSING, SINGLE, COMMENT, PROCESSING} type;
  bool is_comment() { return type==COMMENT;}
  bool is_processing() { return type==PROCESSING;}
  bool is_element() { return !is_comment() && !is_processing();}
};

XMLTag parse_tag(std::istream& p, bool skip_comments = false);

std::string parse_content(std::istream& in);

void skip_element(std::istream& in,const XMLTag&);

void check_tag(std::istream& in, const std::string& name);

} // end namespace alps

#endif // PALM_PARSER_PARSER_H
