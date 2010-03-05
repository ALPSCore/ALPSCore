// Copyright (C) 2008 - 2010 Lukas Gamper <gamperl -at- gmail.com>
// Use, modification and distribution is subject to the Boost Software
// License, Version 1.0. (See accompanying file LICENSE_1_0.txt or copy at
// http://www.boost.org/LICENSE_1_0.txt)


#include <string>
#include <boost/lexical_cast.hpp>
#include <alps/utility/encode.hpp>

namespace alps {
  std::string hdf5_name_encode(std::string const & s)
  {
    std::string r = s;
    char chars[] = {'&', '/'};
    for (std::size_t i = 0; i < sizeof(chars); ++i)
        for (std::size_t pos = r.find_first_of(chars[i]); pos < std::string::npos; pos = r.find_first_of(chars[i], pos + 1))
            r = r.substr(0, pos) + "&#" + boost::lexical_cast<std::string, int>(chars[i]) + ";" + r.substr(pos + 1);
    return r;
  }
  std::string hdf5_name_decode(std::string const & s)
  {
        std::string r = s;
        for (std::size_t pos = r.find_first_of('&'); pos < std::string::npos; pos = r.find_first_of('&', pos + 1))
            r = r.substr(0, pos) + static_cast<char>(boost::lexical_cast<int>(r.substr(pos + 2, r.find_first_of(';', pos) - pos - 2))) + r.substr(r.find_first_of(';', pos) + 1);
        return r;
    }

}
