/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/params.hpp>

#include "gtest/gtest.h"

TEST(param,AccessNonExisting) {
    int argc=1;
    const char* argv[]={"THIS_PROGRAM"};
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

// Extra (undefined) options in command line (not accessed)
TEST(param, ExtraOptions)
{
    // const char* argv[]={"THIS PROGRAM", "--param1=111", "--param2=222"};
    const char* argv[]={"THIS PROGRAM", "--paramABC", "111", "--paramXYZ", "222" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc,argv);

    p.define<int>("paramABC","Integer parameter");
    // p["param2"]=999;
    EXPECT_EQ(111, p["paramABC"]);
}

// Definition of the option after access attempt
TEST(param, DefineAfterAccess)
{
    const char* argv[]={"THIS PROGRAM", "--param", "111"};
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc,argv);

    {
      EXPECT_THROW(int i=p["param"], alps::params::uninitialized_value);
    }
    // After access attempt, the option becomes "explicitly assigned" and cannot be defined
    EXPECT_THROW((p.define<int>("param","Integer parameter")), alps::params::extra_definition);

    // Still, it cannot be read...
    {
      EXPECT_THROW(int i=p["param"], alps::params::uninitialized_value);
    }
    // ...until assigned (any type)
    p["param"]=999.25;
    EXPECT_EQ(999.25, p["param"]);
}

int main(int argc, char **argv) 
{
//    Test();
//    return 0;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

