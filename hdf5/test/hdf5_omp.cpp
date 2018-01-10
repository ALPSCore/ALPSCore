/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <sstream>
#include <cstdio>
#include <alps/hdf5.hpp>

#include "gtest/gtest.h"
#if defined(_OPENMP)
TEST(hdf5, TestingOfOpenMP){
    bool result = false;
    try {
#pragma omp parallel for
        for (unsigned i=0; i<10; ++i) {
            std::stringstream filename;
            filename<<"omp."<<i<<".h5";
            alps::hdf5::archive ar(filename.str(), "w");
            ar["/value"] << i;
        }
        result = true;
    } catch (std::exception & e) {
        std::cerr << "Exception thrown:" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    for (unsigned i=0; i<10; ++i) {
        std::stringstream filename;
        filename<< "omp."<<i<<".h5";
        std::remove(filename.str().c_str());
    }

    EXPECT_TRUE(result);
}
#endif
int main(int argc, char **argv) 
{
#if defined(_OPENMP)
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
#else
    return 0;
#endif
}

