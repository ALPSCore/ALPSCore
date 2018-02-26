/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>

#include <alps/testing/unique_file.hpp>
#include <vector>
#include <complex>

#include "gtest/gtest.h"

using namespace std;

TEST(hdf5, TestingIoOfDoubleVectors){
    alps::testing::unique_file ufile("vvdbl.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();
    {
        vector<vector<double> > v;
        for(int i = 0; i < 3; ++i)
            v.push_back(vector<double>(i+1, 2*i));
        alps::hdf5::archive ar(filename, "w");
        ar["/spectrum/sectors/5/results/cdag-c/mean/value"] = v;
        std::cout << v.size() << std::endl;
    }
    {
        vector<vector<double> > v;
        alps::hdf5::archive ar(filename, "r");
        ar["/spectrum/sectors/5/results/cdag-c/mean/value"] >> v;
        std::cout << v.size() << std::endl;
    }
}
