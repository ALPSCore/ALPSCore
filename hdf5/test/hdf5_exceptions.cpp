/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>

#include <boost/random.hpp>

#include <string>
#include <vector>
#include <iostream>
#include <algorithm>
#include "gtest/gtest.h"
#include <alps/testing/unique_file.hpp>

TEST(hdf5, TestingCopyingOfHDF5){
    alps::testing::unique_file ufile("test_hdf5_exceptions.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();

    {
        alps::hdf5::archive oar(filename, "a");
    }
    {
        using namespace alps;
        alps::hdf5::archive iar(filename, "r");
        double test;
        try {
            iar >> make_pvp("/not/existing/path", test);
        } catch (std::exception& ex) {
            std::string str = ex.what();
            std::size_t start = str.find_first_of("\n");
            std::cout << str.substr(0, start) << std::endl;
        }
    }
}
