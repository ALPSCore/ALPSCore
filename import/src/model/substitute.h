/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
