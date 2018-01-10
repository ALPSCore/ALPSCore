/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_FS_REMOVE_EXTENSIONS_HPP
#define ALPS_UTILITY_FS_REMOVE_EXTENSIONS_HPP

#include <string>

namespace alps { namespace fs {
  /** @brief Returns the file name stem with ALL extensions removed
      @param filename : file name (e.g., /a/b/c.ver1.txt.gz)
      @returns File name stem (e.g., /a/b/c)
  */
  std::string remove_extensions(const std::string& filename);
} }
#endif // ALPS_UTILITY_FS_REMOVE_EXTENSIONS_HPP
