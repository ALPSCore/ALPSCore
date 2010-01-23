// Copyright (C) 2008 - 2010 Lukas Gamper <gamperl -at- gmail.com>
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)

#ifndef ALPS_ESCAPE_HPP
#define ALPS_ESCAPE_HPP

#include <string>
#include <boost/lexical_cast.hpp>
#include <alps/config.h>

namespace alps { 
  ALPS_DECL std::string hdf5_name_encode(std::string const & s);
  ALPS_DECL std::string hdf5_name_decode(std::string const & s);
}
#endif
