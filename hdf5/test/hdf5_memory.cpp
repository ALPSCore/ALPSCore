/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/testing/unique_file.hpp>

#include <iostream>

#include "gtest/gtest.h"
using namespace std;

TEST(hdf5, TestingHDF5Memory){
    alps::testing::unique_file ufile("test_hdf5_memory.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string& memname_h5=ufile.name();
    
    {
        alps::hdf5::archive oa(memname_h5, "w");
        std::vector<std::complex<double> > foo(3);
        std::vector<double> foo2(3);
        oa << alps::make_pvp("/foo", foo);
        oa << alps::make_pvp("/foo2", foo2);
    }
    
    {
 
        std::vector<double> foo, foo2;
        try {
            alps::hdf5::archive ia(memname_h5);
            ia >> alps::make_pvp("/foo", foo);
            ia >> alps::make_pvp("/foo2", foo2);
        } catch (exception e) {
            cout << "Exception caught: no complex value" << endl;
        }
    }
}
