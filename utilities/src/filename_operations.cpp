/*
 * Copyright (C) 1998-2016 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/utilities/remove_extensions.hpp>
#include <alps/utilities/get_basename.hpp>
#include <alps/utilities/get_dirname.hpp>
#include <boost/filesystem.hpp>

namespace alps {
  std::string remove_extensions(const std::string& filename)
  {
      boost::filesystem::path fp=filename;
      while (!fp.extension().empty())  fp.replace_extension();
      return fp.native(); // FIXME: should it be .native() or just .string() ?
  }

  std::string get_basename(const std::string& filename)
  {
      boost::filesystem::path fp=filename;
      return fp.filename().native(); // FIXME: should it be .native() or just .string() ?
  }

  std::string get_dirname(const std::string& filename)
  {
      boost::filesystem::path fp=filename;
      return fp.parent_path().native(); // FIXME: should it be .native() or just .string() ?
  }
}
