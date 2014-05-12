/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010        by Andreas Hehn <hehn@phys.ethz.ch>                   *
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

#include "matrix_unit_tests.hpp"

using alps::numeric::matrix;

BOOST_AUTO_TEST_CASE_TEMPLATE( hdf5, T, test_types )
{
    std::string const filename = "alps_matrix_test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));

    {
        matrix<T> a(10,20);
        resize(a,40,40);
        fill_matrix_with_numbers(a);
        resize(a,10,20);
        matrix<T> b(a);
        b.shrink_to_fit();


        BOOST_CHECK_EQUAL(a.capacity() > b.capacity(), true); // maybe this should be an assert instead

        {
            alps::hdf5::archive ar(filename, alps::hdf5::archive::WRITE | alps::hdf5::archive::REPLACE);
            ar["/matrix"] << a;
        }

        BOOST_CHECK_EQUAL(a,b);
        matrix<T> c;
        alps::hdf5::archive ar2(filename);
        ar2["/matrix"] >> c;

        BOOST_CHECK_EQUAL(a,c);
        BOOST_CHECK_EQUAL(b,c);
    }

    boost::filesystem::remove(boost::filesystem::path(filename));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( hdf5_matrix_matrix, T, test_types )
{
    typedef typename matrix<matrix<T> >::col_element_iterator col_iterator;

    std::string const filename = "alps_matrix_test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));

    {
        matrix<matrix<T> > a(10,20);
        a.resize(40,40);
        T iota = 0;
        for(std::size_t j=0; j < num_cols(a); ++j) {
            for(std::pair<col_iterator,col_iterator> r = col(a,j); r.first != r.second; ++r.first) {
                resize(*r.first,3,5);
                for(std::size_t j=0; j < num_cols(*r.first); ++j) {
                    for(std::pair<typename matrix<T>::col_element_iterator,typename matrix<T>::col_element_iterator> rm = col(*r.first,j); rm.first != rm.second; ++rm.first) {
                        *rm.first = iota;
                        iota += 1;
                    }
                }
            }
        }
        a.resize(10,20);
        matrix<matrix<T> > b(a);
        b.shrink_to_fit();

        BOOST_CHECK_EQUAL(a.capacity() > b.capacity(), true); // maybe this should be an assert instead

        {
            alps::hdf5::archive ar(filename, alps::hdf5::archive::WRITE | alps::hdf5::archive::REPLACE);
            ar["/matrix"] << a;
        }

        BOOST_CHECK_EQUAL(a,b);
        matrix<matrix<T> > c;
        alps::hdf5::archive ar2(filename);
        ar2["/matrix"] >> c;

        BOOST_CHECK_EQUAL(a,c);
        BOOST_CHECK_EQUAL(b,c);
    }

    boost::filesystem::remove(boost::filesystem::path(filename));
}
