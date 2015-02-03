/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/params.hpp>

#include "gtest/gtest.h"

TEST(param,AccessNonExisting) {
    int argc=2;
    const char* argv[]={"THIS_PROGRAM", "/dev/null"};
    alps::params p(argc, argv);

    p.define<int>("defined_par","Defined non-existing parameter");

    {
      EXPECT_THROW(int i=p["defined_par"], alps::params::uninitialized_value);
    }

    const alps::params p1(p);
    {
      EXPECT_THROW(int i=p1["defined_par"], alps::params::uninitialized_value);
    }
}

TEST(param, AccessUndefined){
    alps::params parms;
    parms["hello"]="world";

    EXPECT_EQ(std::string("world"),parms["hello"]);
    {
      EXPECT_THROW(std::string s=parms["not_in_parms"], alps::params::uninitialized_value);
    }
    const alps::params p(parms);

    {
      EXPECT_THROW(std::string s=p["not_in_parms"], alps::params::uninitialized_value);
    }
    EXPECT_EQ(std::string("world"),p["hello"]);
}

int main(int argc, char **argv) 
{
//    Test();
//    return 0;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

