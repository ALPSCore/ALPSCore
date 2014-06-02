/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/params.hpp>
#include "gtest/gtest.h"

TEST(params, TestingOfOrdering){
    std::string const filename = "odering";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    {
        alps::params parms;
        parms["a"] = 6;
        parms["x"] = 2;
        parms["b"] = 3;
        parms["w"] = 1;
        
        for (alps::params::const_iterator it = parms.begin(); it != parms.end(); ++it)
            std::cout << it->first << " " << it->second << std::endl;

        alps::hdf5::archive oar(filename, "w");
        oar["/parameters"] << parms;
    }
    std::cout << "= = = = =" << std::endl;
    {
        alps::params parms;
        alps::hdf5::archive iar(filename, "r");
        iar["/parameters"] >> parms;

        alps::params::const_iterator it = parms.begin();
        assert((it++)->first == "a");
        assert((it++)->first == "x");
        assert((it++)->first == "b");
        assert((it++)->first == "w");
    }
    boost::filesystem::remove(boost::filesystem::path(filename));
}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

