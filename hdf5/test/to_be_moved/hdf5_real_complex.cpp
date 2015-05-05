/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#include <alps/hdf5/archive.hpp>
#include <alps/numeric/matrix.hpp>
#include <alps/utilities/short_print.hpp>

#include <boost/filesystem.hpp>

#include <vector>
#include <complex>
#include <iostream>

template <class T>
std::ostream& operator<<(std::ostream& os, std::vector<T> const& v)
{
	os << "[" <<alps::write_vector(v, " ", 6) << "]";
	return os;
}

int main() {

    if (boost::filesystem::exists("real_complex.h5") && boost::filesystem::is_regular_file("real_complex.h5"))
        boost::filesystem::remove("real_complex.h5");

    try {
        const int vsize = 6, msize=4;

        std::vector<double> v(vsize, 3.2);
        alps::numeric::matrix<double> A(msize,msize, 1.5);

        std::cout << "v: " << v << std::endl;

        {
            alps::hdf5::archive ar("real_complex.h5", "w");
            ar["/matrix"] << A;
            ar["/vec"] << v;
        }

        std::vector<std::complex<double> > w;
        alps::numeric::matrix<std::complex<double> > B;
        {
            alps::hdf5::archive ar("real_complex.h5", "r");
            ar["/matrix"] >> B;
            ar["/vec"] >> w;
        }

        std::cout << "w: " << w << std::endl;
        
        boost::filesystem::remove("real_complex.h5");
        
        return EXIT_FAILURE;

    } catch (alps::hdf5::archive_error) {
        boost::filesystem::remove("real_complex.h5");
        return EXIT_SUCCESS;
    }

}