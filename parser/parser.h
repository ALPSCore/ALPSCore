/***************************************************************************
* ALPS++ library
*
* parser/parser.h   a simple parser 
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
