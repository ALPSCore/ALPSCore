/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_FS_GET_EXTENSION_HPP_f91a77f5077b4f85bc17e9071b47449d
#define ALPS_UTILITY_FS_GET_EXTENSION_HPP_f91a77f5077b4f85bc17e9071b47449d

#include <string>

namespace alps { namespace fs {
  /** @brief Returns the file name extension
      @param filename file name (e.g., "/a/b/c.ver1.txt.gz")
      @returns File name extension (e.g., ".gz")
  */
  std::string get_extension(const std::string& filename);
} }
#endif /* ALPS_UTILITY_FS_GET_EXTENSION_HPP_f91a77f5077b4f85bc17e9071b47449d */
