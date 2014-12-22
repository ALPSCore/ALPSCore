/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/params.hpp>

#include "gtest/gtest.h"
// TODO: make an in-file for all types!
// TODO: make reference output file!

TEST(params, TestingDefaultParamSyntax){

    alps::params parms;
    std::string strg = parms["non_existent_parameter"] | "substitution_string";
    std::cout << strg << std::endl;
}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

