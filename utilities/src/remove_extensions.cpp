/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/utilities/remove_extensions.hpp>
#include <boost/filesystem.hpp>

namespace alps {
  std::string remove_extensions(const std::string& filename)
  {
      boost::filesystem::path fp=filename;
      while (!fp.extension().empty())  fp=fp.stem();
      return fp.native(); // FIXME: should it be .native() or just .string() ?
  }
}
