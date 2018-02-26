/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_FS_GET_BASENAME_HPP_b868097c679f4cd09e0a7f9c3f413406
#define ALPS_UTILITY_FS_GET_BASENAME_HPP_b868097c679f4cd09e0a7f9c3f413406


#include <alps/config.hpp>
#include <string>

namespace alps { namespace fs {
  /** @brief Returns the base name of the file (removing leading directories)
      @param path : pathname (e.g., /a/b/c.ver1.txt.gz)
      @returns File name  (e.g., c.ver1.txt.gz)
  */
  std::string get_basename(const std::string& path);
} }

#endif /* ALPS_UTILITY_FS_GET_BASENAME_HPP_b868097c679f4cd09e0a7f9c3f413406 */
