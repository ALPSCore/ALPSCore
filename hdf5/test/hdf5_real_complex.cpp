/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/utilities/short_print.hpp>
#include <alps/testing/unique_file.hpp>

#include <vector>
#include <complex>
#include <iostream>

#include "gtest/gtest.h"

TEST(hdf5, TestingOfRealComplex){
    alps::testing::unique_file ufile("real_complex.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string& real_complex_h5=ufile.name();
  
    try {
        const int vsize = 6;

        std::vector<double> v(vsize, 3.2);

        std::cout << "v: " << alps::short_print(v) << std::endl;

        {
            alps::hdf5::archive ar(real_complex_h5, "w");
            ar["/vec"] << v;
        }

        std::vector<std::complex<double> > w;
        {
            alps::hdf5::archive ar(real_complex_h5, "r");
            ar["/vec"] >> w;
        }

        std::cout << "w: " << alps::short_print(w) << std::endl;
        
        EXPECT_TRUE(false);

    } catch (alps::hdf5::archive_error) {
        EXPECT_TRUE(true);
    }
}
