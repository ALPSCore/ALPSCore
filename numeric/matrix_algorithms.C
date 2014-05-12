/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010        by Tim Ewart <timothee.ewart@unige.ch>                 *
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

#define BOOST_TEST_SOURCE
#define BOOST_TEST_MODULE alps::numeric::matrix_algorithms

#ifndef ALPS_LINK_BOOST_TEST
#include <boost/test/included/unit_test.hpp>
#else
#include <boost/test/unit_test.hpp>
#endif
#include <boost/filesystem.hpp>
#include <boost/mpl/list.hpp>

#include <boost/preprocessor/repetition.hpp>
#include <boost/preprocessor/arithmetic/add.hpp>
#include <boost/lexical_cast.hpp>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_real_distribution.hpp>
#include <boost/random/variate_generator.hpp>


#include <alps/numeric/matrix.hpp>
#include <alps/numeric/diagonal_matrix.hpp>
#include <alps/numeric/matrix/algorithms.hpp>


#define ValueWG 33
#define tuple1(z, n, unused) BOOST_PP_COMMA_IF(n) size<(BOOST_PP_ADD(n,1)), 11,double> 
#define tuple2(z, n, unused) BOOST_PP_COMMA_IF(n) size<(BOOST_PP_ADD(n,1)),-11,double> 
#define tuple3(z, n, unused) BOOST_PP_COMMA_IF(n) size<(BOOST_PP_ADD(n,1)), 11,std::complex<double> > 
#define tuple4(z, n, unused) BOOST_PP_COMMA_IF(n) size<(BOOST_PP_ADD(n,1)),-11,std::complex<double> > 


template <int n, int m, typename T> // n # of workgroup, T double or std::complex<double> 
struct size {
   BOOST_STATIC_ASSERT(n>0);
   BOOST_STATIC_ASSERT(n*ValueWG > m);
   typedef T value_type; // To template later
   enum {valuex = n*ValueWG+m};// n is the number or work group, m how we resize
   enum {valuey = n*ValueWG-m};// n is the number or work group, m how we resize
};

typedef boost::mpl::list< size<4,0,double>, size<4,0,std::complex<double> >,  BOOST_PP_REPEAT(4,tuple1,~), BOOST_PP_REPEAT(4,tuple2,~), BOOST_PP_REPEAT(4,tuple3,~), BOOST_PP_REPEAT(4,tuple4,~) > test_types;
//typedef boost::mpl::list< size<1,1,double>, size<1,-1,std::complex<double> > > test_types; //Dev line
//typedef boost::mpl::list< size<1,-2,double> > test_types; //Dev line

// Define a base random number generator and initialize it with a seed.
boost::random::mt19937 rng(3); 
// Define distribution U[0,1) [double values]
boost::random::uniform_real_distribution<> dist(0,1);
// Define a random variate generator using our base generator and distribution
boost::variate_generator<boost::random::mt19937&, boost::random::uniform_real_distribution<> > uniDblGen(rng, dist);

using alps::numeric::matrix;
using namespace alps::numeric;

template<typename T>
struct ValidateHelper{
    void static validate(matrix<T> const & M1, matrix<T> const & M2){
        BOOST_CHECK_EQUAL(num_rows(M1),num_rows(M2));
        BOOST_CHECK_EQUAL(num_cols(M1),num_cols(M2));
        for(std::size_t j(0); j< num_cols(M1); ++j)  
            for(std::size_t i(0); i< num_rows(M1); ++i)
                BOOST_CHECK_CLOSE(M1(i,j),M2(i,j),1e-6); 
    };
    void static validateid(matrix<T> const & M1){
        for(std::size_t j(0); j< num_cols(M1); ++j)
            for(std::size_t i(0); i< num_rows(M1); ++i)         
                if (i==j)
                    BOOST_CHECK_CLOSE(M1(i,j),1.0,1e-6);  // checks relative difference
                else
                    BOOST_CHECK_SMALL(M1(i,j),1e-6);      // checks absolute smallness
    };
};

template<typename T>
struct ValidateHelper<std::complex<T> > {
    void static validate(matrix<std::complex<T> > const & M1,matrix<std::complex<T> > const & M2){
        BOOST_CHECK_EQUAL(num_rows(M1),num_rows(M2));
        BOOST_CHECK_EQUAL(num_cols(M1),num_cols(M2));        
        for(std::size_t j(0); j< num_cols(M1); ++j)
            for(std::size_t i(0); i< num_rows(M1); ++i){
                BOOST_CHECK_CLOSE(M1(i,j).real(),M2(i,j).real(),1e-6); 
                BOOST_CHECK_CLOSE(M1(i,j).imag(),M2(i,j).imag(),1e-6); 
            }
    }
    void static validateid(matrix<std::complex<T> > const & M1){ 
        for(std::size_t j(0); j< num_cols(M1); ++j)
            for(std::size_t i(0); i< num_rows(M1); ++i){
                if (i==j)
                    BOOST_CHECK_CLOSE(M1(i,j).real(),1.0,1e-6);
                else
                    BOOST_CHECK_SMALL(M1(i,j).real(),1e-6);
                BOOST_CHECK_SMALL(M1(i,j).imag(),1e-6); 
            }
    }
};

template<typename T>
struct InitHelper{
    void static init(matrix<T> & m){
         std::generate(elements(m).first, elements(m).second, uniDblGen);
    };
};

template<typename T>
struct InitHelper<std::complex<T> > {
    void static init(matrix<std::complex<T> > & M){
        for(std::size_t i(0); i< num_rows(M); ++i) 
            for(std::size_t j(0); j< num_cols(M); ++j){ 
               T r = uniDblGen();
               T m = uniDblGen();
               M(i,j) = std::complex<T>(r, m);
            }
   }
};

BOOST_AUTO_TEST_CASE_TEMPLATE( trace_test, T, test_types)
{
    matrix<typename T::value_type> m(T::valuex, T::valuex);
    InitHelper<typename T::value_type>::init(m);
    typename T::value_type tr = trace(m);
    typename T::value_type check = 0;
    for(std::size_t i=0; i < num_rows(m); ++i)
        check += m(i,i);
    BOOST_CHECK_EQUAL(tr,check);
}
/*---------------------------------------------------------------------- tranpose TESTS */
BOOST_AUTO_TEST_CASE_TEMPLATE(Transpose_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    Matrix M(T::valuex,T::valuey);
    InitHelper<typename T::value_type>::init(M);
    Matrix Mtt(T::valuex,T::valuey);

    Mtt =transpose(M);
    Mtt =transpose(Mtt);
    ValidateHelper<typename T::value_type>::validate(M,Mtt);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Transpose_inplace_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    Matrix M(T::valuex,T::valuey);
    InitHelper<typename T::value_type>::init(M);
    Matrix Mcopy(M);

    transpose_inplace(M);
    transpose_inplace(M);
    ValidateHelper<typename T::value_type>::validate(M,Mcopy);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Adjoint_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    Matrix M(T::valuex,T::valuey);
    InitHelper<typename T::value_type>::init(M);
    Matrix Mtt(T::valuex,T::valuey);

    Mtt = adjoint(adjoint(M));
    ValidateHelper<typename T::value_type>::validate(M,Mtt);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Adjoint_inplace_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    Matrix M(T::valuex,T::valuey);
    InitHelper<typename T::value_type>::init(M);
    Matrix Mcopy(M);

    adjoint_inplace(M);
    adjoint_inplace(M);
    ValidateHelper<typename T::value_type>::validate(M,Mcopy);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Conj_transpose_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    Matrix M(T::valuex,T::valuey);
    InitHelper<typename T::value_type>::init(M);

    Matrix mct = conj(transpose(M));
    Matrix mtc = transpose(conj(M));
    ValidateHelper<typename T::value_type>::validate(mct,mtc);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(Adjoint_conj_transpose_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    Matrix M(T::valuex,T::valuey);
    InitHelper<typename T::value_type>::init(M);

    Matrix ma  = adjoint(M);
    Matrix mtc = transpose(conj(M));
    ValidateHelper<typename T::value_type>::validate(ma,mtc);
}

/*--------------------------------------------------------------------------- SVD TESTS */
BOOST_AUTO_TEST_CASE_TEMPLATE(SVD_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;
    typename associated_real_diagonal_matrix<matrix<typename T::value_type> >::type S;

    Matrix M(T::valuex,T::valuey);
    Matrix U;
    Matrix V;

    InitHelper<typename T::value_type>::init(M);
    svd(M,U,V,S);

    Matrix D(V*adjoint(V)); // complex needs conjugate, no effect on double)
    ValidateHelper<typename T::value_type>::validateid(D);
    Matrix E(adjoint(U)*U); // complex needs conjugate, no effect on double)
    ValidateHelper<typename T::value_type>::validateid(E);
}

/*---------------------------------------------------------------------------- LQ TESTS */
BOOST_AUTO_TEST_CASE_TEMPLATE(LQ_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;

    Matrix M(T::valuex,T::valuey);
    Matrix Q;
    Matrix L;

    InitHelper<typename T::value_type>::init(M);
    lq(M,L,Q);

    Matrix D(L*Q);
    ValidateHelper<typename T::value_type>::validate(M,D);
}

// second test we check Q is an Id matrix, cautions we implemented the thin LQ so only Q*Qt is equal to one
BOOST_AUTO_TEST_CASE_TEMPLATE(LQ_Q_ID_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;

    Matrix M(T::valuex,T::valuey);
    Matrix Q;
    Matrix L;

    InitHelper<typename T::value_type>::init(M);
    lq(M,L,Q);

    Matrix D(Q*adjoint(Q)); // complex needs conjugate, no effect on double)
    ValidateHelper<typename T::value_type>::validateid(D);
}
/*---------------------------------------------------------------------------- QR TESTS */

BOOST_AUTO_TEST_CASE_TEMPLATE(QR_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;

    Matrix M(T::valuex,T::valuey);
    Matrix Q;
    Matrix R;

    InitHelper<typename T::value_type>::init(M);

    qr(M,Q,R);

    Matrix D(Q*R);
    ValidateHelper<typename T::value_type>::validate(M,D);
}

// second test we check Q is an Id matrix, cautions we implemented the thin QR so only Qt*Q is equal to one
BOOST_AUTO_TEST_CASE_TEMPLATE(QR_Q_ID_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;

    Matrix M(T::valuex,T::valuey);
    Matrix Q;
    Matrix R;

    InitHelper<typename T::value_type>::init(M);
    qr(M,Q,R);

    Matrix D(adjoint(Q)*Q); //complex needs conjugate
    ValidateHelper<typename T::value_type>::validateid(D);
}

/*---------------------------------------------------------------------------- Inverse TESTS */
BOOST_AUTO_TEST_CASE_TEMPLATE(Inverse_test, T, test_types)
{
    typedef matrix<typename T::value_type> Matrix;

    Matrix M(T::valuex,T::valuex);

    InitHelper<typename T::value_type>::init(M);
    Matrix invM = inverse(M);

    Matrix D(M*invM);
    ValidateHelper<typename T::value_type>::validateid(D);
}
