/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_FS_GET_DIRNAME_HPP_c9685cbd35144ac3b28ae653a9d0cbc0
#define ALPS_UTILITY_FS_GET_DIRNAME_HPP_c9685cbd35144ac3b28ae653a9d0cbc0

#include <alps/config.hpp>
#include <string>

namespace alps { namespace fs {
  /** @brief Returns the parent direcory name of the file (removing trailing filename)
      @param path : pathname (e.g., /a/b/c.ver1.txt.gz)
      @returns  Parent directory name without the trailing slash (e.g., /a/b) 
  */
  std::string get_dirname(const std::string& path);
} }

#endif /* ALPS_UTILITY_FS_GET_DIRNAME_HPP_c9685cbd35144ac3b28ae653a9d0cbc0 */
