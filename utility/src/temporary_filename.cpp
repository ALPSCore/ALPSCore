/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */


#include <alps/utility/temporary_filename.hpp>
#include <boost/throw_exception.hpp>
#include <boost/filesystem.hpp>


#ifdef BOOST_WINDOWS
#include <fcntl.h>
#include <io.h>
#else
#include <unistd.h>
#endif


namespace alps {
  std::string temporary_filename(std::string name)
  {
    name +="XXXXXX";

    return boost::filesystem::unique_path(name).native();
  }
}
