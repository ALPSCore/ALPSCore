/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2001-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#include <alps/parser/path.h>
#include <alps/config.h>

std::string alps::xslt_path(const std::string& stylefile) {
  std::string path("file:");
  path += ALPS_XML_DIR;
  char* p =getenv("ALPS_XSLT_PATH");
  if (p!=0) 
    path=*p;
  if (path != "http://xml.comp-phys.org" && path != "http://xml.comp-phys.org/")
    return path+"/"+stylefile;
  else if (stylefile == "job.xsl")
    return "http://xml.comp-phys.org/2002/10/job.xsl";
  else if (stylefile == "ALPS.xsl")
    return "http://xml.comp-phys.org/2004/10/ALPS.xsl";
  else if (stylefile == "plot2html.xsl")
    return "http://xml.comp-phys.org/2003/4/plot2html.xsl";
  else
    return "http://xml.comp-phys.org/"+stylefile;
}
  
std::string xml_library_path(const std::string& file) 
{
  return std::string(ALPS_XML_DIR)+"/"+file;
}