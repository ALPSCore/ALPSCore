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

#include <alps/parser/xslt_path.h>
#include <alps/config.h>
#include <alps/version.h>
#include <boost/throw_exception.hpp>
#include <boost/filesystem/operations.hpp>
#include <stdexcept>

std::string alps::xslt_path(const std::string& stylefile) {
  char* p = getenv("ALPS_XML_PATH");
  if (p==0)
    return stylefile == "job.xsl" || stylefile == "plot2html.xsl" ? "ALPS.xsl" : stylefile;
  
  std::string path = p;
  
  if (path != "http://xml.comp-phys.org" && path != "http://xml.comp-phys.org/")
    return path+"/"+stylefile;
  else if (stylefile == "ALPS.xsl")
    return "http://xml.comp-phys.org/2009/12/ALPS.xsl";
  else
    return "http://xml.comp-phys.org/"+stylefile;
}


std::string alps::search_xml_library_path(const std::string& file)
{
  // check whether this is a full path and the file exists 
  boost::filesystem::path path(file);
  if (boost::filesystem::exists(path))
    return path.string();
    
  char* p = getenv("ALPS_XML_PATH");
  if (p != 0)
    path = boost::filesystem::path(p) /  file;
  else { 
    p = getenv("ALPS_ROOT");
    if (p != 0) 
      path = boost::filesystem::path(p) / "lib" / "xml" / file;
    else {
      path = boost::filesystem::path(ALPS_XML_DIR) / file;
    #ifdef ALPS_XML_ALTERNATE_DIR
      if (!boost::filesystem::exists(path))
        path = boost::filesystem::path(ALPS_XML_ALTERNATE_DIR) / file;
    #endif
    }
  }
  if (!boost::filesystem::exists(path))
    boost::throw_exception(std::runtime_error("Cannot find file " + file + ", failed to find it at " + path.string()));
  return path.string();
}

void alps::copy_stylesheet(boost::filesystem::path const& dir)
{
  boost::filesystem::path dest = dir / "ALPS.xsl";
  if (! boost::filesystem::exists(dest))
    boost::filesystem::copy_file(search_xml_library_path("ALPS.xsl"),dest);
}

