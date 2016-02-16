/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_UTILITY_TEMPORARY_FILENAME_HPP
#define ALPS_UTILITY_TEMPORARY_FILENAME_HPP

#include <alps/config.hpp>
#include <string>

namespace alps {
  /** @brief Generates a random file name with a given prefix
      @param prefix : the file prefix
  */
  std::string temporary_filename(std::string prefix);
}
#endif // ALPS_UTILITY_TEMPORARY_FILENAME_HPP
