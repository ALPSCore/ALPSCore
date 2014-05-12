/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013        by Michele Dolfi <dolfim@phys.ethz.ch>,               *
 *                              Andreas Hehn <hehn@phys.ethz.ch>                   *
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

#include <iostream>
#include <boost/filesystem.hpp>
#include <alps/numeric/matrix.hpp>
#include <alps/version.h>
#include <alps/hdf5.hpp>


int main() {
    boost::filesystem::path   infile(ALPS_SRCDIR);
    infile = infile / "test" / "numeric" / "matrix_deprecated_hdf5_format_test.h5";
    if (!boost::filesystem::exists(infile))
    {
        std::cout << "Reference file " << infile << " not found." << std::endl;
        return -1;
    }
    alps::numeric::matrix<double> m;
    alps::hdf5::archive ar(infile.native(), "r");
    ar["/matrix_old_hdf5_format"] >> m;
    std::cout << "Matrix " << num_rows(m) << "x" << num_cols(m) << ":\n";
    std::cout << "capacity: " <<m.capacity().first << "x" << m.capacity().second <<"\n";
    std::cout << "data:\n" << m;
    return 0;
}
