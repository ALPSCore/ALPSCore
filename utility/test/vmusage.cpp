/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utility/vmusage.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <iostream>

int main(int argc, char** argv) {
  int pid = (argc == 1) ? -1 : boost::lexical_cast<int>(argv[1]);
  BOOST_FOREACH(alps::vmusage_type::value_type v, alps::vmusage(pid)) {
    std::cerr << v.first << " = " << v.second << "\n";
  }
  return 0;
}

  
