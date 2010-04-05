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

/// \file path.h
/// \brief functions to obtain the path to ALPS XSLT and XML files

#ifndef ALPS_PARSER_PATH_H
#define ALPS_PARSER_PATH_H

#include <alps/config.h>
#include <boost/filesystem/path.hpp>
#include <cstdlib>
#include <string>

namespace alps {

/// \brief given the name of an XSLT file, return the full path
///
/// The default behavior is to just return the file name
///
/// If the environment variable ALPS_XSLT_PATH is set, the contents of
/// ALPS_XSLT_PATH are used instead of the default path.
///
/// A special case is if http://xml.comp-phys.org/ is used as the
/// value of ALPS_XSLT_PATH. In that case not the string returned
/// points to the version of the XSLT file valid at the time of
/// release of the library. E.g.  given the file name "ALPS.xsl" the
/// returned string might be
/// "http://xml.comp-phys.org/2004/10/ALPS.xsl".
extern ALPS_DECL std::string xslt_path(const std::string& stylefile);

/// \brief returns the full path to the specified XML file and checks whether the file exists.
///
/// The function prepends the path to the ALPS XML/XSLT library
/// directory to the specified filename.  \throw \c std::runtime_error
/// if the file does not exist in the ALPS XML/XSLT library directory.
extern ALPS_DECL std::string search_xml_library_path(const std::string& file);
  
/// \brief copies the ALPS.xsl stylesheet to the specifeid directory
///
/// This function copies the ALPS.xsl stylesheet to the specified directory.
/// The function does not overwrite an already existing file with the name ALPS.xsl

extern ALPS_DECL void copy_stylesheet(boost::filesystem::path const& dir);

} // end namespace alps

#endif // ALPS_PARSER_PARSER_H
