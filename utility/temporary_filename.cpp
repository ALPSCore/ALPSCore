// Copyright (C) 2008 - 2010 Lukas Gamper <gamperl -at- gmail.com>
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)


#include <alps/utility/temporary_filename.hpp>
#include <boost/throw_exception.hpp>
#include <boost/lexical_cast.hpp>
#include <stdexcept>
#include <sys/stat.h>
#include <iostream>
#include <cstdlib>


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

#ifdef BOOST_MSVC
    name = _mktemp(const_cast<char*>(name.c_str()));
    int res=0; 
    //int res=open(name.c_str(),O_RDWR|O_BINARY|O_CREAT|O_EXCL|_O_SHORT_LIVED, 128|256);
#else
    name = mktemp(const_cast<char*>(name.c_str()));
    int res=0;
    //int res = mkstemp(const_cast<char*>(name.c_str()));
#endif
    if (res<0)
      boost::throw_exception(std::runtime_error("Could not open temporary file"));
    return name;
  }
}
