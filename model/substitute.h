/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2005 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_SUBSTITUTE_H
#define ALPS_MODEL_SUBSTITUTE_H

#include <boost/algorithm/string/replace.hpp>

namespace alps {
  
inline std::string substitute(std::string const& text, unsigned int type)
{
  std::string n;
  for (unsigned int i=0;i<text.size();++i)
  if (text[i]=='#')
    n += boost::lexical_cast<std::string>(type);
  else
    n += text[i];
  // std::cerr << "Replaced " << text << " to " << n << " by substituting " << boost::lexical_cast<std::string>(type) << "\n";
  return n;
//  return boost::algorithm::replace_all_copy(text,"#",boost::lexical_cast<std::string>(type));
}
  
inline Parameters substitute(Parameters const& parms, unsigned int type)
{
  Parameters p;
  for (Parameters::const_iterator it = parms.begin() ; it != parms.end(); ++it)
    p[substitute(it->key(),type)] = substitute(it->value(),type);
  return p;
}

} // namespace alps

#endif
