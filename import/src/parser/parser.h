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

/// \file parser/parser.h
/// \brief a very simple XML parser

#ifndef ALPS_PARSER_PARSER_H
#define ALPS_PARSER_PARSER_H

#include <alps/config.h>
#include <alps/cctype.h>
#include <alps/parser/xmlattributes.h>
#include <map>
#include <string>

namespace alps {

namespace detail {

// \brief returns true if the character is a XML tag name
inline bool is_identifier_char(char c) 
{ 
  return std::isalnum(c) || c=='_' || c==':' || c=='#';
}

} // end namespace detail

/// \brief a struct to store the contents of an XML tag
struct XMLTag
{
  /// the name of the tag
  std::string name;
  /// the attributes
  XMLAttributes attributes;
/*/// \brief the type of tag
  ///
  /// The meaning of the values, and what is stored in the \c name member can be seen in the following table
  /// \htmlonly
  /// <TABLE BORDER=2>
  /// <TR><TD><B>type</B></TD><TD><B>name</B></TD><TD><B>example</B></TD></TR>
  /// <TR><TD><TT>OPENING</TT></TD><TD><TT>TAG</TT></TD><TD><TT>&lt;TAG&gt;</TT></TD></TR>
  /// <TR><TD><TT>CLOSING</TT></TD><TD><TT>/TAG</TT></TD><TD><TT>&lt;/TAG&gt;</TT></TD></TR>
  /// <TR><TD><TT>SINGLE</TT></TD><TD><TT>TAG</TT></TD><TD><TT>&lt;TAG/&gt;</TT></TD></TR>
  /// <TR><TD><TT>COMMENT</TT></TD><TD><TT>!</TT></TD><TD><TT>&lt;!-- comment --!/&gt;</TT></TD></TR>
  /// <TR><TD><TT>PROCESSING</TT></TD><TD><TT>?</TT></TD><TD><TT>&lt;? processing instruction ?/&gt;</TT></TD></TR>
  /// </TABLE>
  /// \endhtmlonly
  /// \latexonly
  /// \begin{tabulate}[|c|c|c|]
  /// \hline
  /// type & name & example \\ %
  /// \hline
  /// {\tt OPENING}    & {\tt TAG}  & {\tt <TAG>} \\  %
  /// {\tt CLOSING}    & {\tt /TAG} & {\tt </TAG>} \\  %
  /// {\tt SINGLE}     & {\tt TAG}  & {\tt <TAG/>} \\  %
  /// {\tt COMMENT}    & {\tt !}    & {\tt <!-- comment --!/>} \\ %
  /// {\tt PROCESSING} & {\tt !}    & {\tt <? processing instruction ?/>} \\ %
  /// \hline
  /// \end{tabulate}  
  /// \endlatexonly
*/enum {OPENING, CLOSING, SINGLE, COMMENT, PROCESSING} type;
  /// returns true if the tag is a comment
  bool is_comment() { return type==COMMENT;}
  /// returns true if the tag is a processing instruction
  bool is_processing() { return type==PROCESSING;}
  /// returns true if the tag is an opening or closing tag of an XML element
  bool is_element() { return !is_comment() && !is_processing();}
};


/// reads an XML tag or attribute name from a \c std::istream
std::string parse_identifier(std::istream& in);

/// \brief reads an ALPS parameter name from a \c std::istream
/// 
/// valid characters, in addition to those in an XML identifier are \c ', 
/// and additionally any arbitrary sequence of characters (including whitespace) surrounded by \c [ ... \ ] 
/// characters, such as in \c MEASURE[Staggered \c Magnetization^2] .
ALPS_DECL std::string parse_parameter_name(std::istream& in);

/// \brief reads until the next occurence of the character \a end or until the end of the stream is reached. 
///
/// \param in the stream to be read
/// \param end the character until which should be read
/// \return  string containing the characters read, excluding leading and trailing whitespace 
/// and excluding the terminating character \a end.
std::string read_until(std::istream& in, char end);

/// \brief checks that the next character read from the stream.
/// \param in the stream to be read
/// \param c the character that should be read
/// \param err the error message to be used if the next character is not \a c.
/// \throw \c std::runtime_error( \a err \c ) if the next character is not \a c
///  reads the next character (slipping white space) and checks if it is the same
///  as the character passed as argument \a c and throws a \c std::runtime_error otherwise.
ALPS_DECL void check_character(std::istream& in, char c, const std::string& err);

/// \brief parses an XML tag
/// \param in the stream to be read
/// \param skip_comments if true, the function skips any comments or processing instructions while parsing
/// \return an \c XMLTag structure containing information about the tag
ALPS_DECL XMLTag parse_tag(std::istream& in, bool skip_comments = true);

/// reads the contents of an element, until the first < character found
ALPS_DECL std::string parse_content(std::istream& in);

/// \brief skips an XML element
/// \param in the stream to be read
/// \param tag the opening tag of the element to be skipped
/// the function reads until it finds the closing tag correesponding to the \a tag passed as argument.
ALPS_DECL void skip_element(std::istream& in,const XMLTag& tag);

/// \brief checks whether the next tag in the XML file has the given name
/// \param in the stream to be read
/// \param name the name of the expected tag
/// \throw \c std::runtime_error if the next tag read from the stream \a in does not have the name given by the argument \a name.
ALPS_DECL void check_tag(std::istream& in, const std::string& name);

} // end namespace alps

#endif // ALPS_PARSER_PARSER_H
