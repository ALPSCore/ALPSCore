/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include "alps/params.hpp"

#include "gtest/gtest.h"

#include <iostream>

TEST(param,Redefinition) {
  // int argc=4;
  // const char* argv[]={"THIS_PROGRAM","--help","--param=1234","--other-param=1234"};
  int argc=2;
  const char* argv[]={"THIS_PROGRAM","--param=1234"};
  alps::params p(argc,argv);
  p.description("Test program");

  p.define<int>("param,p1",10,"Integer parameter");
  p.define<int>("param,p2",20,"The same integer parameter");
  p.define<double>("param,p3",30.0,"The same parameter as double");

  // p.help_requested(std::cerr); // throws too

  EXPECT_THROW((p["param"]),boost::program_options::ambiguous_option);

  // EXPECT_THROW((p["param"]),boost::program_options::ambiguous_option);
  // EXPECT_EQ(p["param"], 1234);
}

int main(int argc, char **argv) 
{
    // Test();
    // return 0;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
