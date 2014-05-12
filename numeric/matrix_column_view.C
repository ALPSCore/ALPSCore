/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
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
#include <alps/numeric/matrix/column_view.hpp>

using alps::numeric::matrix;
using alps::numeric::vector;
using alps::numeric::column_view;

BOOST_AUTO_TEST_CASE_TEMPLATE( size_test, T, test_types )
{
    matrix<T> c(15,5);
    column_view<matrix<T> > cv(c,3);
    BOOST_CHECK_EQUAL(cv.size(),15);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( const_element_access, T, test_types )
{
    using std::distance;
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    column_view<matrix<T> > const cv(c,3);
    for(std::size_t i = 0; i < num_rows(c); ++i)
    {
        BOOST_CHECK_EQUAL( c(i,3), cv(i));
        BOOST_CHECK_EQUAL( c(i,3), cv[i]);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( const_iterator_test, T, test_types )
{
    using std::distance;
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    column_view<matrix<T> > const cv(c,3);
    for(typename column_view<matrix<T> >::const_iterator it = cv.begin(); it != cv.end(); ++it)
        BOOST_CHECK_EQUAL( c(distance(cv.begin(),it),3), *it);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( iterator_test, T, test_types )
{
    using std::distance;
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    for(typename column_view<matrix<T> >::iterator it = cv.begin(); it != cv.end(); ++it)
        *it = 100*distance(cv.begin(),it);
    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL(c(i,5),T(100*i));

    for(std::size_t j=0; j < num_cols(c); ++j)
    {
        if(j != 5)
            for(std::size_t i = 0; i < num_rows(c); ++i)
                BOOST_CHECK_EQUAL(d(i,j),c(i,j));
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( element_assign_test, T, test_types )
{
    using std::distance;
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    for(std::size_t i = 0; i < num_rows(c); ++i)
        cv(i) = 100*i;

    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL(c(i,5),T(100*i));

    for(std::size_t i = 0; i < num_rows(c); ++i)
        cv[i] = 1000*i;

    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL(c(i,5),T(1000*i));

    for(std::size_t j=0; j < num_cols(c); ++j)
    {
        if(j != 5)
            for(std::size_t i = 0; i < num_rows(c); ++i)
                BOOST_CHECK_EQUAL(d(i,j),c(i,j));
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( conversion_to_vector, T, test_types)
{
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    vector<T> v(cv);
    BOOST_CHECK_EQUAL(num_rows(c),v.size());
    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL(c(i,5),v(i));
    BOOST_CHECK_EQUAL(c,d);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( plus_assign, T, test_types)
{
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    vector<T> v(15);
    fill_range_with_numbers(v.begin(),v.end(),0);
    cv += v;
    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL( c(i,5), d(i,5)+v[i]);

    for(std::size_t j=0; j < num_cols(c); ++j)
    {
        if(j != 5)
            for(std::size_t i = 0; i < num_rows(c); ++i)
                BOOST_CHECK_EQUAL(d(i,j),c(i,j));
    }

    cv += cv;

    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL( c(i,5), T(2)*(d(i,5)+v[i]));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( minus_assign, T, test_types)
{
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    vector<T> v(15);
    fill_range_with_numbers(v.begin(),v.end(),0);
    cv -= v;
    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL( c(i,5), d(i,5)-v[i]);

    for(std::size_t j=0; j < num_cols(c); ++j)
    {
        if(j != 5)
            for(std::size_t i = 0; i < num_rows(c); ++i)
                BOOST_CHECK_EQUAL(d(i,j),c(i,j));
    }

    cv -= cv;

    for(std::size_t i = 0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL( c(i,5), T(0));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( multiply_assign, T, test_types)
{
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,19);

    cv *= T(5);

    for(std::size_t i=0; i < num_rows(c); ++i)
        BOOST_CHECK_EQUAL(c(i,19), T(5)*d(i,19));

    for(std::size_t j=0; j < num_cols(c); ++j)
    {
        if(j != 19)
            for(std::size_t i = 0; i < num_rows(c); ++i)
                BOOST_CHECK_EQUAL(c(i,j),d(i,j));
        else
            for(std::size_t i=0; i < num_rows(c); ++i)
                BOOST_CHECK_EQUAL(c(i,19), T(5)*d(i,19));
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scalar_product_column_view_vector, T, test_types)
{
    using alps::numeric::conj;
    // We assume conj works properly
    assert( conj(std::complex<double>(1,2)) == std::complex<double>(1,-2) );
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    vector<T> v(15);
    fill_range_with_numbers(v.begin(),v.end(),0);

    T r = scalar_product(cv, v);

    T ref(0);

    for(std::size_t i=0; i < v.size(); ++i)
        ref += conj(d(i,5))*v[i];

    BOOST_CHECK_EQUAL(ref, r);

    T r2 = scalar_product(v, cv);

    BOOST_CHECK_EQUAL(conj(ref) , r2);
    BOOST_CHECK_EQUAL(d,c);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( scalar_product_column_view_column_view, T, test_types)
{
    using alps::numeric::conj;
    // We assume conj works properly
    assert( conj(std::complex<double>(1,2)) == std::complex<double>(1,-2) );
    matrix<T> c(15,20);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);
    column_view<matrix<T> > cv(c,5);
    vector<T> v(15);
    fill_range_with_numbers(v.begin(),v.end(),0);

    T r = scalar_product(cv, cv);

    T ref(0);

    for(std::size_t i=0; i < v.size(); ++i)
        ref += conj(d(i,5))*d(i,5);

    BOOST_CHECK_EQUAL(ref, r);
    BOOST_CHECK_EQUAL(d,c);
}

