/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_UTIL_H
#define PARAPACK_UTIL_H

#include <alps/config.h>
#include <string>

namespace alps {

int hash(int n, int s = 826);

std::string id2string(int id, std::string const& pad = "_");

ALPS_DECL double parse_percentage(std::string const& str);

} // end namespace alps

#endif // PARAPACK_UTIL_H
