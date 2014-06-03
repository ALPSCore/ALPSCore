/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/hdf5/vector.hpp>
#include <alps/hdf5/complex.hpp>
#include <alps/utility/vectorio.hpp>

#include <boost/filesystem.hpp>

#include <vector>
#include <complex>
#include <iostream>
#include "gtest/gtest.h"

template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& v)
{
	os << "[" <<alps::write_vector(v, " ", 6) << "]";
	return os;
}

TEST(hdf5, TestingOfRealComplex){

    if (boost::filesystem::exists("real_complex_vec.h5") && boost::filesystem::is_regular_file("real_complex_vec.h5"))
        boost::filesystem::remove("real_complex_vec.h5");

    try {
        const int size = 6;

        std::vector<double> v(size, 3.2);

        std::cout << "v: " << v << std::endl;

        {
            alps::hdf5::archive ar("real_complex_vec.h5", "w");
            ar["/vec"] << v;
        }

        std::vector<std::complex<double> > w;
        {
            alps::hdf5::archive ar("real_complex_vec.h5", "r");
            ar["/vec"] >> w;
        }

        std::cout << "w: " << w << std::endl;
        
        boost::filesystem::remove("real_complex_vec.h5");
        
		bool passed = true;
		for (int i=0; passed && i<size; ++i)
			passed = (v[i] == w[i]);

        std::cout << "Test status checked element by element." << std::endl;
        EXPECT_TRUE(passed);

    } catch (alps::hdf5::archive_error) {
        boost::filesystem::remove("real_complex_vec.h5");
        std::cout << "Test passed because Exception was thrown." << std::endl;
    }

}
int main(int argc, char **argv) 
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

