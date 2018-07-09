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

// Test that double vector can NOT be read as complex
TEST(hdf5, WriteDoubleReadComplexVec) {
    alps::testing::unique_file ufile("real_complex_vec.h5.", alps::testing::unique_file::REMOVE_NOW);
    const std::string&  filename = ufile.name();

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
        ASSERT_THROW(ar["/vec"] >> w, std::runtime_error);
    }
}
