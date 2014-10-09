/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

// Return virtual memory usage information (works only for Linux)
  
#ifndef ALPS_UTILITY_VMUSAGE_HPP
#define ALPS_UTILITY_VMUSAGE_HPP

#include <alps/config.hpp>
#include <map>
#include <string>


namespace alps {

typedef std::map<std::string, unsigned long> vmusage_type;

ALPS_DECL vmusage_type vmusage(int pid = -1);

} // end namespace alps

#endif // ALPS_UTILITY_VMUSAGE_HPP
