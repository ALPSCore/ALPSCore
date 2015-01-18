/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/params.hpp>

#include "gtest/gtest.h"

TEST(params, TestingMissingParameter){
    alps::params parms;
    parms["hello"]="world";

    EXPECT_EQ(std::string("world"),parms["hello"].as<std::string>());
    EXPECT_THROW(parms["not_in_parms"].as<std::string>(),boost::bad_any_cast);
    
    const alps::params p(parms);

    EXPECT_THROW(p["not_in_parms"].as<std::string>(), boost::bad_any_cast);
}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

