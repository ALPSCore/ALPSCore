/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/utility/vmusage.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/foreach.hpp>
#include <iostream>

#include "gtest/gtest.h"

TEST(vmusage, main)
{
    std::string test = "my_test";
    int pid = boost::lexical_cast<int>(test.c_str());
    BOOST_FOREACH(alps::vmusage_type::value_type v, alps::vmusage(pid)) {
        std::cerr << v.first << " = " << v.second << "\n";
        };
}

int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
 
