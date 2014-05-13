/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Michele Dolfi <dolfim@phys.ethz.ch>                *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

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