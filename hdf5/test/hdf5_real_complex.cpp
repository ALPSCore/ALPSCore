/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
// TODO: do we have a matrix class test?
// #include <alps/numeric/matrix.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/filesystem.hpp>

#include <vector>
#include <complex>
#include <iostream>

#include "gtest/gtest.h"

TEST(hdf5, TestingOfRealComplex){

    if (boost::filesystem::exists("real_complex.h5") && boost::filesystem::is_regular_file("real_complex.h5"))
        boost::filesystem::remove("real_complex.h5");

    try {
        const int vsize = 6, msize=4;

        std::vector<double> v(vsize, 3.2);
        // alps::numeric::matrix<double> A(msize,msize, 1.5);

        std::cout << "v: " << alps::short_print(v) << std::endl;

        {
            alps::hdf5::archive ar("real_complex.h5", "w");
            // ar["/matrix"] << A;
            ar["/vec"] << v;
        }

        std::vector<std::complex<double> > w;
        // alps::numeric::matrix<std::complex<double> > B;
        {
            alps::hdf5::archive ar("real_complex.h5", "r");
            // ar["/matrix"] >> B;
            ar["/vec"] >> w;
        }

        std::cout << "w: " << alps::short_print(w) << std::endl;
        
        boost::filesystem::remove("real_complex.h5");
        
        EXPECT_TRUE(false);

    } catch (alps::hdf5::archive_error) {
        boost::filesystem::remove("real_complex.h5");
        EXPECT_TRUE(true);
    }
}
