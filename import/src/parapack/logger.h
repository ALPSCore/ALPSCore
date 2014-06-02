/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_LOGGER_H
#define PARAPACK_LOGGER_H

#include <alps/parapack/process.h>
#include <alps/parapack/types.h>
#include <alps/utility/vmusage.hpp>
#include <iostream>
#include <string>

namespace alps {

struct logger {
  static std::string header();
  static std::string task(alps::tid_t tid);
  static std::string clone(alps::tid_t tid, alps::cid_t cid);
  static std::string group(alps::process_group g);
  static std::string group(alps::thread_group g);
  static std::string usage(alps::vmusage_type const& u);
};

} // namespace alps

#endif // PARAPACK_LOGGER_H
