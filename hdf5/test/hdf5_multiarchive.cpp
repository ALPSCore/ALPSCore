/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>

#include <boost/filesystem.hpp>
#include <boost/random.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "gtest/gtest.h"

TEST(hdf5, TestingOfMultiArchive){
    std::string const filename = "test_hdf5_multiarchive.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    {
        using namespace alps;
        alps::hdf5::archive oar(filename, "a");
    }
    {
        using namespace alps;
        alps::hdf5::archive iar(filename);
        alps::hdf5::archive oar(filename, "a");
        oar << make_pvp("/data", 42);
    }
    {
        using namespace alps;
        alps::hdf5::archive iar(filename, "r");
        int test;
        iar >> make_pvp("/data", test);
        {
            alps::hdf5::archive iar2(filename, "r");
            int test2;
            iar2 >> make_pvp("/data", test2);
            iar >> make_pvp("/data", test);
        }
        iar >> make_pvp("/data", test);
        {
            alps::hdf5::archive iar3(filename, "r");
            int test3;
            iar >> make_pvp("/data", test);
            iar3 >> make_pvp("/data", test3);
        }
        iar >> make_pvp("/data", test);
    }
    {
        using namespace alps;
        alps::hdf5::archive iar4(filename, "r");
        int test4;
        iar4 >> make_pvp("/data", test4);
    }
    boost::filesystem::remove(boost::filesystem::path(filename));
}
