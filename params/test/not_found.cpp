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
    
    EXPECT_THROW(p["defined_par"].as<int>(), boost::bad_any_cast);

    const alps::params p1(p);
    EXPECT_THROW(p1["defined_par"].as<int>(), boost::bad_any_cast);
}

TEST(param, AccessUndefined){
    alps::params parms;
    parms["hello"]="world";

    EXPECT_EQ(std::string("world"),parms["hello"].as<std::string>());
    EXPECT_THROW(parms["not_in_parms"].as<std::string>(),boost::bad_any_cast);
    
    const alps::params p(parms);

    EXPECT_THROW(p["not_in_parms"].as<std::string>(), boost::bad_any_cast);
    EXPECT_EQ(std::string("world"),p["hello"].as<std::string>());
}

int main(int argc, char **argv) 
{
//    Test();
//    return 0;
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

