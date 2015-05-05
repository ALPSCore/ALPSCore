/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/matrix.hpp>

#include <boost/filesystem.hpp>

#include <vector>
#include <complex>
#include <iostream>

int main() {

    if (boost::filesystem::exists("real_complex_matrix.h5") && boost::filesystem::is_regular_file("real_complex_matrix.h5"))
        boost::filesystem::remove("real_complex_matrix.h5");

    try {
        const int size=4;

        alps::numeric::matrix<double> A(size,size, 1.5);

        {
            alps::hdf5::archive ar("real_complex_matrix.h5", "w");
            ar["/matrix"] << A;
        }

        alps::numeric::matrix<std::complex<double> > B;
        {
            alps::hdf5::archive ar("real_complex_matrix.h5", "r");
            ar["/matrix"] >> B;
        }
        
        boost::filesystem::remove("real_complex_matrix.h5");
        
        bool passed = true;
        for (int i=0; passed && i<size; ++i)
            for (int j=0; passed && j<size; ++j)
                passed = (A(i,j) == B(i,j));
        
        std::cout << "Test status checked element by element." << std::endl;
        return (passed) ? EXIT_SUCCESS : EXIT_FAILURE;

    } catch (alps::hdf5::archive_error) {
        boost::filesystem::remove("real_complex_matrix.h5");
        std::cout << "Test passed because Exception was thrown." << std::endl;
        return EXIT_SUCCESS;
    }

}