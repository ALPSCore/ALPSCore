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

TEST(hdf5, TestingOfRealComplexVec){
    alps::testing::unique_file ufile("real_complex_vec.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();

    try {
        const int size = 6;

        std::vector<double> v(size, 3.2);

        std::cout << "v: " << alps::short_print(v) << std::endl;

        {
            alps::hdf5::archive ar(filename, "w");
            ar["/vec"] << v;
        }

        std::vector<std::complex<double> > w;
        {
            alps::hdf5::archive ar(filename, "r");
            ar["/vec"] >> w;
        }

        std::cout << "w: " << alps::short_print(w) << std::endl;
        
		bool passed = true;
		for (int i=0; passed && i<size; ++i)
			passed = (v[i] == w[i]);

        std::cout << "Test status checked element by element." << std::endl;
        EXPECT_TRUE(passed);

    } catch (alps::hdf5::archive_error & ex) {
        std::cout << "Test passed because Exception was thrown." << std::endl;
        EXPECT_TRUE(true);
    }

}
