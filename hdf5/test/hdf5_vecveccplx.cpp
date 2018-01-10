/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/testing/unique_file.hpp>
#include <vector>
#include <complex>

#include "gtest/gtest.h"

using namespace std;

TEST(hdf5, TestingIoOfComplexVectors){
    alps::testing::unique_file ufile("vvcplx.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();
    {
        vector< vector< complex<double> > > v;
        for( int i = 0; i < 3; ++i )
            v.push_back(vector< complex<double> >(i+1, complex<double>(i,2*i)));
        alps::hdf5::archive ar(filename,"w");
        ar << alps::make_pvp("v",v);
    }
}

