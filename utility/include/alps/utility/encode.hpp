/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_ESCAPE_HPP
#define ALPS_ESCAPE_HPP

#include <string>
#include <alps/config.h>

namespace alps { 
  ALPS_DECL std::string hdf5_name_encode(std::string const & s);
  ALPS_DECL std::string hdf5_name_decode(std::string const & s);
}
#endif
