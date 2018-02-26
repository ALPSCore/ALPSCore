/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/testing/unique_file.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "gtest/gtest.h"

TEST(hdf5, TestingOfMultiArchive){
    alps::testing::unique_file ufile("test_hdf5_multiarchive.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string& filename=ufile.name();
    
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
}
