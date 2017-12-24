/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPSCORE_COMMON_HPP
#define ALPSCORE_COMMON_HPP

namespace alps {
  // define function return type and return statement based on the value that should be returned
  #define DECLTYPE(ret) decltype(ret) { return ret; }
}

#endif //ALPSCORE_COMMON_HPP
