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

#define BOOST_TEST_MODULE alps::numeric::matrix

#include <boost/test/included/unit_test.hpp>
#include <boost/filesystem.hpp>
#include <boost/mpl/list.hpp>

#include <boost/lambda/lambda.hpp>
#include <complex>
#include <numeric>
#include <iostream>

#include <alps/numeric/matrix.hpp>
#include <alps/numeric/matrix/matrix_interface.hpp>
#include <alps/numeric/matrix/vector.hpp>

//
// List of types T for which the matrix<T> is tested
//
typedef boost::mpl::list<
      float
    , double
    , int
    , unsigned int
    , long unsigned int
    , std::complex<float>
    , std::complex<double>
> test_types;
// long long unsigned int causes problems in boost::iterator facade


namespace type_pairs
{
struct DComplexDouble
{
    typedef std::complex<double> first_type;
    typedef double second_type;
};

struct DoubleDComplex
{
    typedef double first_type;
    typedef std::complex<double> second_type;
};

struct IntDouble
{
    typedef int first_type;
    typedef double second_type;
};

struct DoubleInt
{
    typedef double first_type;
    typedef int second_type;
};
};

//
// List of type pairs <T,U> for which the mixed type matrix std::vector multiplication is tested.
//
typedef boost::mpl::list<type_pairs::IntDouble, type_pairs::DoubleInt, type_pairs::DoubleDComplex, type_pairs::DComplexDouble> test_type_pairs;

template <typename OutputIterator, typename T>
T fill_range_with_numbers(OutputIterator begin, OutputIterator end, T iota)
{
    // Unfortunately we can't use the postincrement operator, due to std:complex<>
    // -> so we have to emulate it's behaviour...
    std::transform(begin,end,begin,boost::lambda::_1 = (boost::lambda::var(iota)+=T(1))-T(1));
    return iota;
}

template <typename T>
T fill_matrix_with_numbers(alps::numeric::matrix<T>& a)
{
    T iota(0);
    for(unsigned int i=0; i<num_rows(a); ++i)
    {
        std::pair<typename alps::numeric::matrix<T>::row_element_iterator, typename alps::numeric::matrix<T>::row_element_iterator> range(row(a,i));
        iota += fill_range_with_numbers(range.first,range.second,T(i));
    }
    return iota;
}

using alps::numeric::matrix;

BOOST_AUTO_TEST_CASE_TEMPLATE( constructors_test, T, test_types )
{
    matrix<T> a;
    BOOST_CHECK_EQUAL(num_rows(a), 0 );
    BOOST_CHECK_EQUAL(num_cols(a), 0 );

    matrix<T> b(10,10);
    BOOST_CHECK_EQUAL(num_rows(b), 10 );
    BOOST_CHECK_EQUAL(num_cols(b), 10 );
    for(unsigned int i=0; i<10; ++i)
        for(unsigned int j=0; j<10; ++j)
            BOOST_CHECK_EQUAL(b(i,j), T());

    matrix<T> c(15,5,5);
    BOOST_CHECK_EQUAL(num_rows(c), 15 );
    BOOST_CHECK_EQUAL(num_cols(c), 5 );
    for(unsigned int i=0; i<15; ++i)
        for(unsigned int j=0; j<5; ++j)
            BOOST_CHECK_EQUAL(c(i,j), T(5));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( column_constructors_test, T, test_types )
{
    std::size_t num_of_cols = 20;
    std::size_t num_of_rows = 30;
    std::vector<std::vector<T> > original(num_of_cols,std::vector<T>(num_of_rows) );
    T iota = 1;
    for(std::size_t i=0; i < original.size(); ++i)
        iota = fill_range_with_numbers(original[i].begin(),original[i].end(),iota);

    typedef typename std::vector<T>::iterator iterator;
    std::vector<std::pair<iterator,iterator> > columns;
    for(std::size_t i=0; i < original.size(); ++i)
        columns.push_back(std::make_pair(original[i].begin(),original[i].end()));

    matrix<T> a(columns);

    for(std::size_t j=0; j < num_of_cols; ++j)
        for(std::size_t i=0; i < num_of_rows; ++i)
            BOOST_CHECK_EQUAL(a(i,j),original[j][i]);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( copy_swap_test, T, test_types )
{
    matrix<T> a(10,10,1);
    matrix<T> b(1,1,0);
    matrix<T> c(a);
    matrix<T> d(b);
    swap(a,b);
    BOOST_CHECK_EQUAL(a,d);
    BOOST_CHECK_EQUAL(b,c);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( assignement_test, T, test_types )
{
    matrix<T> a(10,10,1);
    matrix<T> b(1,1,0);
    b = a;
    BOOST_CHECK_EQUAL(a,b);
    b(0,0) = 5;
    BOOST_CHECK_EQUAL(a(0,0) != b(0,0), true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( row_iterator_test, T, test_types )
{
    matrix<T> a(10,20);
    fill_matrix_with_numbers(a);

    for(unsigned int i=0; i<num_rows(a); ++i)
    {
        std::pair<typename matrix<T>::row_element_iterator, typename matrix<T>::row_element_iterator> range(row(a,i));
        unsigned int j=0;
        for(typename matrix<T>::const_row_element_iterator it(range.first); it != range.second; ++it)
        {
            BOOST_CHECK_EQUAL(a(i,j), *it);
            ++j;
        }
    }
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
            BOOST_CHECK_EQUAL(a(i,j),T(i+j));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( col_iterator_test, T, test_types )
{
    matrix<T> a(10,20);
    fill_matrix_with_numbers(a);
    for(unsigned int j=0; j<num_cols(a); ++j)
    {
        std::pair<typename matrix<T>::col_element_iterator, typename matrix<T>::col_element_iterator> range(col(a,j));
        unsigned int i=0;
        for(typename matrix<T>::const_col_element_iterator it(range.first); it != range.second; ++it)
        {
            BOOST_CHECK_EQUAL(a(i,j), *it);
            ++i;
        }
    }
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
            BOOST_CHECK_EQUAL(a(i,j),T(i+j));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( element_iterator_test, T, test_types )
{
    matrix<T> a(10,20);
    matrix<T> b(10,20);
    std::pair<typename matrix<T>::element_iterator,typename matrix<T>::element_iterator> range(elements(a));
    fill_range_with_numbers(range.first,range.second,0);

    T k = T(0);
    T sum = T(0);
    for(unsigned int j=0; j<num_cols(a); ++j)
        for(unsigned int i=0; i<num_rows(a); ++i)
        {
            b(i,j) = k;
            sum += k;
            k += T(1);
        }

    T acc = std::accumulate(range.first, range.second,T(0));
    BOOST_CHECK_EQUAL(acc,sum);
    BOOST_CHECK_EQUAL(a,b);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( resize_test, T, test_types )
{
    matrix<T> a;

    // Check primitive enlargement
    resize(a,10,5);
    BOOST_CHECK_EQUAL(num_rows(a),10);
    BOOST_CHECK_EQUAL(num_cols(a),5);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    // Resize case 1:
    // Enlargement out of the reserved range
    // size1 > reserved_size1_
    unsigned int size1 = a.capacity().first + 10;
    // Check whether enlargement keeps the values of the original matrix
    resize(a,size1,15,1);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if( i >=10 || j >= 5)
                BOOST_CHECK_EQUAL(a(i,j),T(1));
            else
                BOOST_CHECK_EQUAL(a(i,j),T(i+j));
        }

    // Resize case 2:
    // Shrinking
    // size1 < reserved_size1
    // size1 < size1_ (-> shrinking)
    resize(a,10,5);
    BOOST_CHECK_EQUAL(a,b);

    // Resize case 3:
    // Enlargement within the already reserved range
    // size1 < reserved_size1
    // size1 > size1_
    resize(a,15,10);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if( i >= 10 || j >= 5) BOOST_CHECK_EQUAL(a(i,j),T(0));
            else BOOST_CHECK_EQUAL(a(i,j), T(i+j));
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( resize_exception_test, T, test_types )
{
    matrix<T> a(22,18);
    fill_matrix_with_numbers(a);

    // What happens if an exception is thrown?
    // Remains the matrix unchanged if an exception is thrown during the resize process?
    // Case 1: size1 > reserved_size1_
    matrix<T> ref(a);
    matrix<T> c(a);
    matrix<T> d(a);
    std::vector<T> test;
    std::size_t max_size = test.max_size();
    try
    {
        resize(a,max_size+10,1);
    }
    catch(...)
    {
        BOOST_CHECK_EQUAL(a,ref);
    }

    // Resize case 2:
    // Shrinking in one dimension
    // size1 < reserved_size1
    // size1 < size1_ (-> shrinking)
    try
    {
        resize(c,1,max_size+10);
    }
    catch(...)
    {
        BOOST_CHECK_EQUAL(c,ref);
    }

    // Resize case 3:
    // Enlargement within the already reserved range
    // size1 < reserved_size1
    // size1 > size1_
    resize(d,2,5);
    matrix<T> ref_d(d);
    try
    {
        resize(d,4,max_size/2+5);
    }
    catch(...)
    {
        BOOST_CHECK_EQUAL(d,ref_d);
    }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( reserve_test, T, test_types)
{
    matrix<T> a(22,18);
    fill_matrix_with_numbers(a);
    
    matrix<T> ref(a);

    // Case 1:
    // size1 > reserved_size1_
    a.reserve(30,30);
    BOOST_CHECK_EQUAL(a,ref);
    BOOST_CHECK_EQUAL(a.capacity().first >= 30 && a.capacity().second >= 30, true);
    
    // Case 2:
    // size1 < reserved_size1_
    // reserved_size1_*size2 > values_.capacity

    a.reserve(20,40);
    BOOST_CHECK_EQUAL(a,ref);
    BOOST_CHECK_EQUAL(a.capacity().first >= 30 && a.capacity().second >= 40, true);


    // Case 3:
    // size1 < reserved_size1_
    // size2 < size2_
    a.reserve(10,10);
    BOOST_CHECK_EQUAL(a,ref);
    BOOST_CHECK_EQUAL(a.capacity().first >= 30 && a.capacity().second >= 40, true);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( append_rows_test, T, test_types)
{
    const unsigned int initsize = 20;
    matrix<T> a(initsize,initsize);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    std::vector<T> data_single(initsize,1);
    std::vector<T> data_multiple(3*initsize,2);
    T iota(0);
    iota = fill_range_with_numbers(data_single.begin(),data_single.end(),iota);
    iota = fill_range_with_numbers(data_multiple.begin(),data_multiple.end(),iota);

    // Append a single row
    append_rows(a, std::make_pair(data_single.begin(), data_single.end()) );
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if( i != initsize)
                BOOST_CHECK_EQUAL(a(i,j),b(i,j));
            else
                BOOST_CHECK_EQUAL(a(i,j),T(j));
        }
    // Append multiple rows
    append_rows(a, std::make_pair(data_multiple.begin(),data_multiple.end()),3);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if( i < initsize)
                BOOST_CHECK_EQUAL(a(i,j),b(i,j));
            else
            {
                switch (i)
                {
                    case initsize:
                        BOOST_CHECK_EQUAL(a(i,j),T(j));
                        break;
                    case initsize+1:
                        BOOST_CHECK_EQUAL(a(i,j),T(j+initsize));
                        break;
                    case initsize+2:
                        BOOST_CHECK_EQUAL(a(i,j),T(j+2*initsize));
                        break;
                    case initsize+3:
                        BOOST_CHECK_EQUAL(a(i,j),T(j+3*initsize));
                        break;
                    default:
                        // There should not be any other row
                        // Report an error
                        BOOST_CHECK( true == false);
                }
            }

        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( append_cols_test, T, test_types)
{
    const unsigned int initsize = 20;
    matrix<T> a(initsize,initsize);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    std::vector<T> data_single(initsize,1);
    std::vector<T> data_multiple(3*initsize,2);
    T iota(0);
    iota = fill_range_with_numbers(data_single.begin(),data_single.end(),iota);
    iota = fill_range_with_numbers(data_multiple.begin(),data_multiple.end(),iota);

    // Append a single column
    append_cols(a, std::make_pair(data_single.begin(), data_single.end()) );
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if( j != initsize)
                BOOST_CHECK_EQUAL(a(i,j),b(i,j));
            else
                BOOST_CHECK_EQUAL(a(i,j),T(i));
        }
    // Append multiple rows
    append_cols(a, std::make_pair(data_multiple.begin(),data_multiple.end()),3);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if( j < initsize)
                BOOST_CHECK_EQUAL(a(i,j),b(i,j));
            else
            {
                switch (j)
                {
                    case initsize:
                        BOOST_CHECK_EQUAL(a(i,j),T(i));
                        break;
                    case initsize+1:
                        BOOST_CHECK_EQUAL(a(i,j),T(i+initsize));
                        break;
                    case initsize+2:
                        BOOST_CHECK_EQUAL(a(i,j),T(i+2*initsize));
                        break;
                    case initsize+3:
                        BOOST_CHECK_EQUAL(a(i,j),T(i+3*initsize));
                        break;
                    default:
                        // There should not be any other column
                        // Report an error
                        BOOST_CHECK( true == false);
                }
            }
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( remove_rows_test, T, test_types)
{
    const unsigned int initsize = 20;
    matrix<T> a(initsize,initsize);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    // remove the last row
    remove_rows(a,initsize-1);
    // remove the first row
    remove_rows(a,0);
    //remove some rows in the middle
    remove_rows(a,5);
    remove_rows(a,11,4);

    BOOST_CHECK_EQUAL(num_rows(a),initsize-7);

    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if(i<5)
                BOOST_CHECK_EQUAL(a(i,j),b(i+1,j));
            else if (i < 11)
                BOOST_CHECK_EQUAL(a(i,j),b(i+2,j));
            else
                BOOST_CHECK_EQUAL(a(i,j),b(i+6,j));
        }
    
    matrix<T> c(b);

}

BOOST_AUTO_TEST_CASE_TEMPLATE( remove_cols_test, T, test_types)
{
    const unsigned int initsize = 20;
    matrix<T> a(initsize,initsize);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    // remove the last row
    remove_cols(a,initsize-1);
    // remove the first row
    remove_cols(a,0);
    //remove some cols in the middle
    remove_cols(a,5);
    remove_cols(a,11,4);

    BOOST_CHECK_EQUAL(num_cols(a),initsize-7);

    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            if(j<5)
                BOOST_CHECK_EQUAL(a(i,j),b(i,j+1));
            else if (j < 11)
                BOOST_CHECK_EQUAL(a(i,j),b(i,j+2));
            else
                BOOST_CHECK_EQUAL(a(i,j),b(i,j+6));
        }
    
    matrix<T> c(b);

}

BOOST_AUTO_TEST_CASE_TEMPLATE( insert_rows_test, T, test_types)
{
    const unsigned int initsize = 20;
    matrix<T> a(initsize,initsize);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    std::vector<T> data_single(20,1);
    std::vector<T> data_multiple(3*initsize,2);
    T iota(0);
    iota = fill_range_with_numbers(data_single.begin(),data_single.end(),iota);
    iota = fill_range_with_numbers(data_multiple.begin(),data_multiple.end(),iota);

    // Insert a row in for the 0th line, the last line and in the middle
    insert_rows(a, initsize, std::make_pair(data_single.begin(), data_single.end()) );
    insert_rows(a, 0, std::make_pair(data_single.begin(), data_single.end()) );
    insert_rows(a, 5, std::make_pair(data_single.begin(), data_single.end()) );
    insert_rows(a, 8, std::make_pair(data_multiple.begin(),data_multiple.end()),3);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            switch(i)
            {
                case 0:
                case 5:
                case 25:
                    BOOST_CHECK_EQUAL(a(i,j),T(j));
                    break;
                case 8:
                case 9:
                case 10:
                    BOOST_CHECK_EQUAL(a(i,j),T(j+(i-7)*initsize));
                    break;
                default:
                    if( i>10 )
                        BOOST_CHECK_EQUAL(a(i,j),b(i-5,j));
                    else if( i>5 )
                        BOOST_CHECK_EQUAL(a(i,j),b(i-2,j));
                    else
                        BOOST_CHECK_EQUAL(a(i,j),b(i-1,j));
            }
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( insert_cols_test, T, test_types)
{ 
    const unsigned int initsize = 20;
    matrix<T> a(initsize,initsize);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    std::vector<T> data_single(20,1);
    std::vector<T> data_multiple(3*initsize,2);
    T iota(0);
    iota = fill_range_with_numbers(data_single.begin(),data_single.end(),iota);
    iota = fill_range_with_numbers(data_multiple.begin(),data_multiple.end(),iota);
    
    // Insert a column in for the 0th line, the last line and in the middle
    insert_cols(a, initsize, std::make_pair(data_single.begin(), data_single.end()) );
    insert_cols(a, 0, std::make_pair(data_single.begin(), data_single.end()) );
    insert_cols(a, 5, std::make_pair(data_single.begin(), data_single.end()) );
    insert_cols(a, 8, std::make_pair(data_multiple.begin(),data_multiple.end()),3);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
        {
            switch(j)
            {
                case 0:
                case 5:
                case 25:
                    BOOST_CHECK_EQUAL(a(i,j),T(i));
                    break;
                case 8:
                case 9:
                case 10:
                    BOOST_CHECK_EQUAL(a(i,j),T(i+(j-7)*initsize));
                    break;
                default:
                    if( j>10 )
                        BOOST_CHECK_EQUAL(a(i,j),b(i,j-5));
                    else if( j>5 )
                        BOOST_CHECK_EQUAL(a(i,j),b(i,j-2));
                    else
                        BOOST_CHECK_EQUAL(a(i,j),b(i,j-1));
            }
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( plus_assign_test, T, test_types)
{
    matrix<T> a(20,30);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);
    
    a += b;
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
            BOOST_CHECK_EQUAL( a(i,j), T((i+j)*2) );

    a += a;
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
            BOOST_CHECK_EQUAL( a(i,j), T((i+j)*4) );
}

BOOST_AUTO_TEST_CASE_TEMPLATE( minus_assign_test, T, test_types)
{
    matrix<T> a(20,30);
    matrix<T> zero(20,30,T(0));
    fill_matrix_with_numbers(a);
    matrix<T> b(a);
    a += b;
    a -= b;
    BOOST_CHECK_EQUAL(a,b);
    
    a -= a;
    BOOST_CHECK_EQUAL(a,zero);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( multiplies_assign_test, T, test_types)
{
    matrix<T> a(20,30);
    matrix<T> zero(20,30,T(0));
    fill_matrix_with_numbers(a);
    matrix<T> b(a);
    a *= T(1);
    BOOST_CHECK_EQUAL(a,b);
    a *= T(0);
    BOOST_CHECK_EQUAL(a,zero);

    fill_matrix_with_numbers(a);
    a *= T(2);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
            BOOST_CHECK_EQUAL( a(i,j), T(i+j)*T(2) );
}

BOOST_AUTO_TEST_CASE_TEMPLATE( plus_test, T, test_types)
{
    matrix<T> a(20,30);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);

    matrix<T> c = a + b;
    a +=b;
    BOOST_CHECK_EQUAL(c,a);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( minus_test, T, test_types)
{
    matrix<T> a(20,30);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);
    a += b;
    matrix<T> c = a - b;
    BOOST_CHECK_EQUAL(c,b);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( multiplies_test, T, test_types)
{
    matrix<T> a(20,30);
    fill_matrix_with_numbers(a);
    matrix<T> b(a);
    matrix<T> ref_b(b);
    a*= T(2);
    matrix<T> c = T(2) * b;
    //TODO Do we really want to assume commutative types?
    matrix<T> d = b * T(2);
    BOOST_CHECK_EQUAL(c,a);
    BOOST_CHECK_EQUAL(d,a);
    BOOST_CHECK_EQUAL(b,ref_b);

    // Check whether or not it works with mixed types.
    // (value_type != T2 ) - at least for non integer types...
    matrix<T> e(b);
    b*= 5;
    for(unsigned int i=0; i<num_rows(c); ++i)
        for(unsigned int j=0; j<num_cols(c); ++j)
        {
            typename matrix<T>::value_type tmp (e(i,j));
            tmp *= 5;
            BOOST_CHECK_EQUAL(b(i,j),tmp);
        }
    matrix<T> ref_e(e);
    matrix<T> f ( e * 5 );
    matrix<T> g ( 5 * e );
    BOOST_CHECK_EQUAL(b,f);
    BOOST_CHECK_EQUAL(b,g);
    BOOST_CHECK_EQUAL(ref_e,e);

}

// BOOST_AUTO_TEST_CASE_TEMPLATE( matrix_vector_multiply_test, T, test_types)
// {
//     alps::numeric::matrix<T> a(20,30);
//     std::vector<T> v(30);
//     fill_matrix_with_numbers(a);
//     fill_range_with_numbers(v.begin(),v.end(),T(0));
//     alps::numeric::matrix<T> a_(a);
//     std::vector<T> v_(v);
//     
//     std::vector<T> result(a*v);
//     BOOST_CHECK_EQUAL(result.size(),num_rows(a));
//     BOOST_CHECK_EQUAL(a,a_);
//     BOOST_CHECK_EQUAL(v,v_);
//     for(unsigned int i=0; i<num_rows(a); ++i)
//     {
//         T row_result(0);
//         for(unsigned int j=0; j<num_cols(a); ++j)
//             row_result += a(i,j)*v(j);
//         BOOST_CHECK_EQUAL(result(i),row_result);
//     }
// 
// }

// BOOST_AUTO_TEST_CASE_TEMPLATE( matrix_vector_multiply_mixed_types_test, TPair, test_type_pairs)
// {
//     // -alps::numeric::matrix<T> * std::vector<int>
//     
//     alps::numeric::matrix<typename TPair::first_type> a(20,30);
//     std::vector<typename TPair::second_type> v(30);
//     fill_matrix_with_numbers(a);
//     fill_range_with_numbers(v.begin(),v.end(),0);
//     alps::numeric::matrix<typename TPair::first_type> a_(a);
//     std::vector<typename TPair::second_type> v_(v);
//     
//     std::vector<typename alps::numeric::MultiplyReturnType<typename TPair::first_type,std::vector<typename TPair::first_type>,typename TPair::second_type, std::vector<typename TPair::second_type> >::value_type> result(a*v);
//     BOOST_CHECK_EQUAL(result.size(),num_rows(a));
//     BOOST_CHECK_EQUAL(a,a_);
//     BOOST_CHECK_EQUAL(v,v_);
//     for(unsigned int i=0; i<num_rows(a); ++i)
//     {
//         typename alps::numeric::MultiplyReturnType<typename TPair::first_type, std::vector<typename TPair::first_type>,typename TPair::second_type, std::vector<typename TPair::second_type> >::value_type row_result(0);
//         for(unsigned int j=0; j<num_cols(a); ++j)
//             row_result += a(i,j)*v(j);
//         BOOST_CHECK_EQUAL(result(i),row_result);
//     }
// }

BOOST_AUTO_TEST_CASE_TEMPLATE( matrix_matrix_multiply_test, T, test_types)
{
    matrix<T> a(20,30);
    matrix<T> b(30,50);
    fill_matrix_with_numbers(a);
    fill_matrix_with_numbers(b);

    matrix<T> c = a * b;

    BOOST_CHECK_EQUAL(num_rows(c), num_rows(a));
    BOOST_CHECK_EQUAL(num_cols(c), num_cols(b));

    for(unsigned int i=0; i<num_rows(c); ++i)
        for(unsigned int j=0; j<num_cols(c); ++j)
        {
            T result(0);
            for(unsigned int k=0; k< num_cols(a); ++k)
                result += a(i,k) * b(k,j);
            BOOST_CHECK_EQUAL(c(i,j),result);
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( conjugate_test, T, test_types )
{
    using alps::numeric::conj;
    matrix<T> a(10,20);
    fill_matrix_with_numbers(a);

    matrix<T> b(a);
    conj_inplace(a);
    for(unsigned int i=0; i<num_rows(a); ++i)
        for(unsigned int j=0; j<num_cols(a); ++j)
            BOOST_CHECK_EQUAL(a(i,j),conj(b(i,j)));

    matrix<T> c(conj(a));

    BOOST_CHECK_EQUAL(c,b);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( transpose_test, T, test_types )
{
    matrix<T> a(30,20);
    fill_matrix_with_numbers(a);

    matrix<T> b;

    b = transpose(a);

    matrix<T> const c(a);

    matrix<T> d = transpose(c);

    for(unsigned int j=0; j < num_cols(a); ++j)
        for(unsigned int i=0; i < num_rows(a); ++i)
        {
            BOOST_CHECK_EQUAL(a(i,j),b(j,i));
            BOOST_CHECK_EQUAL(c(i,j),d(j,i));
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( transpose_inplace_squared_test, T, test_types )
{
    matrix<T> a(30,30);
    fill_matrix_with_numbers(a);

    matrix<T> b(a);

    transpose_inplace(b);

    for(unsigned int j=0; j < num_cols(a); ++j)
        for(unsigned int i=0; i < num_rows(a); ++i)
            BOOST_CHECK_EQUAL(a(i,j),b(j,i));
}

BOOST_AUTO_TEST_CASE_TEMPLATE( transpose_inplace_test, T, test_types )
{
    matrix<T> a(30,10);
    fill_matrix_with_numbers(a);

    matrix<T> b(a);

    transpose_inplace(b);

    for(unsigned int j=0; j < num_cols(a); ++j)
        for(unsigned int i=0; i < num_rows(a); ++i)
            BOOST_CHECK_EQUAL(a(i,j),b(j,i));

    matrix<T> c(10,30);
    fill_matrix_with_numbers(c);
    matrix<T> d(c);

    transpose_inplace(d);
    for(unsigned int j=0; j < num_cols(c); ++j)
        for(unsigned int i=0; i < num_rows(c); ++i)
            BOOST_CHECK_EQUAL(c(i,j),d(j,i));
}

/* Doesn't work yet. 
BOOST_AUTO_TEST_CASE_TEMPLATE( transpose_plus_test, T, test_types )
{
    matrix<T> a(30,20);
    matrix<T> b(20,30);
    fill_matrix_with_numbers(a);
    fill_matrix_with_numbers(b);

    matrix<T> c = transpose(a) + b;

    for(unsigned int j=0; j < num_cols(c); ++j)
        for(unsigned int i=0; i < num_rows(c); ++i)
            BOOST_CHECK_EQUAL(c(i,j), a(j,i)*b(i,j));
}
*/
BOOST_AUTO_TEST_CASE_TEMPLATE( transpose_matrix_matrix_multiply_test, T, test_types )
{
    matrix<T> a(30,20);
    matrix<T> b(30,50);
    fill_matrix_with_numbers(a);
    fill_matrix_with_numbers(b);

    matrix<T> c = transpose(a) * b;

    BOOST_CHECK_EQUAL(num_rows(c), num_cols(a));
    BOOST_CHECK_EQUAL(num_cols(c), num_cols(b));

    for(unsigned int i=0; i<num_rows(c); ++i)
        for(unsigned int j=0; j<num_cols(c); ++j)
        {
            T result(0);
            for(unsigned int k=0; k< num_rows(a); ++k)
                result += a(k,i) * b(k,j);
            BOOST_CHECK_EQUAL(c(i,j),result);
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( matrix_transpose_matrix_multiply_test, T, test_types )
{
    matrix<T> a(20,30);
    matrix<T> b(50,30);
    fill_matrix_with_numbers(a);
    fill_matrix_with_numbers(b);

    matrix<T> c = a * transpose(b);

    BOOST_CHECK_EQUAL(num_rows(c), num_rows(a));
    BOOST_CHECK_EQUAL(num_cols(c), num_rows(b));

    for(unsigned int i=0; i<num_rows(c); ++i)
        for(unsigned int j=0; j<num_cols(c); ++j)
        {
            T result(0);
            for(unsigned int k=0; k< num_cols(a); ++k)
                result += a(i,k) * b(j,k);
            BOOST_CHECK_EQUAL(c(i,j),result);
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( transpose_transpose_multiply_test, T, test_types)
{
    matrix<T> a(50,60);
    matrix<T> b(40,50);
    fill_matrix_with_numbers(a);
    fill_matrix_with_numbers(b);

    matrix<T> c = transpose(a) * transpose(b);

    BOOST_CHECK_EQUAL(num_rows(c), num_cols(a));
    BOOST_CHECK_EQUAL(num_cols(c), num_rows(b));

    for(unsigned int i=0; i<num_rows(c); ++i)
        for(unsigned int j=0; j<num_cols(c); ++j)
        {
            T result(0);
            for(unsigned int k=0; k< num_rows(a); ++k)
                result += a(k,i) * b(j,k);
            BOOST_CHECK_EQUAL(c(i,j),result);
        }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( hdf5, T, test_types )
{
    std::string const filename = "alps_matrix_test.h5";
    if (boost::filesystem::exists(boost::filesystem::path(filename)))
        boost::filesystem::remove(boost::filesystem::path(filename));
    matrix<T> a(10,20);
    resize(a,40,40);
    fill_matrix_with_numbers(a);
    resize(a,10,20);
    matrix<T> b(a);

    BOOST_CHECK_EQUAL(a.capacity() > b.capacity(), true); // maybe this should be an assert instead

    {
        alps::hdf5::archive ar(filename, alps::hdf5::archive::WRITE | alps::hdf5::archive::REPLACE);
        ar << alps::make_pvp("/matrix",a);
    }

    BOOST_CHECK_EQUAL(a,b);
    matrix<T> c;
    alps::hdf5::archive ar2(filename);
    ar2 >> alps::make_pvp("/matrix",c);

    BOOST_CHECK_EQUAL(a,c);
    BOOST_CHECK_EQUAL(b,c);
    boost::filesystem::remove(boost::filesystem::path(filename));
}
