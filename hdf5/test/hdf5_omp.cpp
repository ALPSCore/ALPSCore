/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <iostream>
#include <alps/hdf5.hpp>

#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>
#include "gtest/gtest.h"
#if defined(_OPENMP)
TEST(hdf5, TestingOfOpenMP){
    bool result = false;
    try {
#pragma omp parallel for
        for (unsigned i=0; i<10; ++i) {
            std::string filename = "omp." + boost::lexical_cast<std::string>(i) + ".h5";
            alps::hdf5::archive ar(filename, "w");
            ar["/value"] << i;
        }
        result = true;
    } catch (std::exception & e) {
        std::cerr << "Exception thrown:" << std::endl;
        std::cerr << e.what() << std::endl;
    }
    for (unsigned i=0; i<10; ++i) {
        std::string filename = "omp." + boost::lexical_cast<std::string>(i) + ".h5";
        if (boost::filesystem::exists(boost::filesystem::path(filename)))
            boost::filesystem::remove(boost::filesystem::path(filename));
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

