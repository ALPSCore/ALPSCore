/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/params.hpp>

#include "gtest/gtest.h"

TEST(params, TestingMissingPramaeter){
    alps::params parms;
    parms["hello"]="world";

    try {
        std::cout<<parms["hello"]<<std::endl;
        std::cout<<parms["not_in_parms"]<<std::endl;
    } catch (std::exception const & e) {
        std::string w = e.what();
        std::cout << w.substr(0, w.find_first_of('\n')) << std::endl;
    }
    
    const alps::params p(parms);

    try {
        std::cout<<p["not_in_parms"]<<std::endl;
    } catch (std::exception const & e) {
        std::string w = e.what();
        std::cout << w.substr(0, w.find_first_of('\n')) << std::endl;
    }

}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

