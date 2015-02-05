/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <vector>
#include "alps/params.hpp"
#include "gtest/gtest.h"

// Test interaction with boost::program_options --- short options are prohibited
TEST(param, ProgramOptions)
{
    const char* argv[]={ "THIS PROGRAM", "--param=123" };
    const int argc=sizeof(argv)/sizeof(*argv);

    alps::params p(argc, argv);

    EXPECT_THROW((p.define<int>("param,p", "Int parameter")), alps::params::invalid_name);

    // if allowed boost::po style of option names as "long,short", these 2 below would crash on assertion failure.
    { EXPECT_THROW(int i=p["param"], alps::params::uninitialized_value); }
    { EXPECT_THROW(int i=p["p"], alps::params::uninitialized_value); } 
}


int main(int argc, char **argv) 
{
    // return Test(),0;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

