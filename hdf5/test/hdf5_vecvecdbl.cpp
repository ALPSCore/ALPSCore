/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>

#include <boost/filesystem.hpp>

#include <vector>
#include <complex>

#include "gtest/gtest.h"

using namespace std;

TEST(hdf5, TestingIoOfDoubleVectors){
    if (boost::filesystem::exists(boost::filesystem::path("vvdbl.h5")))
        boost::filesystem::remove(boost::filesystem::path("vvdbl.h5"));
    {
        vector<vector<double> > v;
        for(int i = 0; i < 3; ++i)
            v.push_back(vector<double>(i+1, 2*i));
        alps::hdf5::archive ar("vvdbl.h5", "w");
        ar["/spectrum/sectors/5/results/cdag-c/mean/value"] = v;
        std::cout << v.size() << std::endl;
    }
	 {
        vector<vector<double> > v;
        alps::hdf5::archive ar("vvdbl.h5", "r");
        ar["/spectrum/sectors/5/results/cdag-c/mean/value"] >> v;
        std::cout << v.size() << std::endl;
    }
    boost::filesystem::remove(boost::filesystem::path("vvdbl.h5"));
}
