/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Alex Kosenkov <alex.kosenkov@gmail.com>            *
 *                              Andreas Hehn <hehn@phys.ethz.ch>                   *
 *                              Michele Dolfi <dolfim@phys.ethz.ch>                *
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

#ifndef ALPS_MATRIX_ALGORITHMS_HPP
#define ALPS_MATRIX_ALGORITHMS_HPP
#include <vector>
#include <stdexcept>
#include <alps/numeric/matrix/matrix_concept_check.hpp>

#include <boost/numeric/bindings/lapack/driver/gesvd.hpp>
#include <boost/numeric/bindings/lapack/driver/gesdd.hpp>
#include <boost/numeric/bindings/lapack/driver/syevd.hpp>
#include <boost/numeric/bindings/lapack/driver/heevd.hpp>
#include <boost/numeric/bindings/lapack/computational/geqrf.hpp>
#include <boost/numeric/bindings/std/vector.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <alps/numeric/real.hpp>
#include <alps/numeric/imag.hpp>
#include <alps/numeric/conj.hpp>

// forward declaration for nested specialization, be cautious of the namespace

namespace alps {
    namespace numeric {
        template <class T, class MemoryBlock>
        class matrix;
    }
}

namespace alps {
    namespace numeric {

        template <typename Matrix>
        Matrix adjoint(Matrix const& m)
        {
            BOOST_CONCEPT_ASSERT((alps::numeric::Matrix<Matrix>));
            // TODO: perhaps this could return a proxy object
            Matrix tmp(num_cols(m), num_rows(m));
            for(typename Matrix::size_type i=0; i < num_rows(m); ++i){
                for(typename Matrix::size_type j=0; j < num_cols(m); ++j){
                    tmp(j,i) = conj(m(i,j));
                }
            }
            return tmp;
        }

        template <typename Matrix>
        void adjoint_inplace(Matrix & m)
        {
            BOOST_CONCEPT_ASSERT((alps::numeric::Matrix<Matrix>));
            // TODO: perhaps this could return a proxy object
            Matrix tmp(num_cols(m), num_rows(m));
            for(typename Matrix::size_type i=0; i < num_rows(m); ++i){
                for(typename Matrix::size_type j=0; j < num_cols(m); ++j){
                    tmp(j,i) = conj(m(i,j));
                }
            }
            swap(tmp, m);
        }

        template <typename Matrix>
        typename Matrix::value_type trace(Matrix const& m)
        {
            BOOST_CONCEPT_ASSERT((alps::numeric::Matrix<Matrix>));
            assert(num_rows(m) == num_cols(m));
            typename Matrix::value_type tr(m(0,0));
            for(typename Matrix::size_type i = 1; i<num_rows(m); ++i)
                tr += m(i,i);
            return tr;
        }

        template<class Matrix>
        Matrix identity_matrix(typename Matrix::size_type size)
        {
            return Matrix::identity_matrix(size);
        }

        template<class Matrix>
        Matrix direct_sum(Matrix const & a, Matrix const & b)
        {
            Matrix ret(num_rows(a)+num_rows(b), num_cols(a)+num_cols(b));
            typedef typename Matrix::size_type st;
            for (st r = 0; r < num_rows(a); ++r)
                for (st c = 0; c < num_cols(a); ++c)
                    ret(r, c) = a(r, c);
            for (st r = 0; r < num_rows(b); ++r)
                for (st c = 0; c < num_cols(b); ++c)
                    ret(r+num_rows(a), c+num_cols(a)) = b(r, c);
            return ret;
        }

        template <typename T>
        typename matrix<T>::value_type norm_square(const matrix<T>& M){
            typename matrix<T>::value_type ret(0);
            for (std::size_t c = 0; c < num_cols(M); ++c)
                for (std::size_t r = 0; r < num_rows(M); ++r)
                    ret += conj(M(r,c)) * M(r,c);
            return ret;
        }

        template <typename T>
        typename matrix<T>::value_type overlap(const matrix<T> & M1, const matrix<T> & M2){
            typename matrix<T>::value_type ret(0);
            for (std::size_t c = 0; c < num_cols(M1); ++c)
                for (std::size_t r = 0; r < num_rows(M1); ++r)
                    ret += conj(M1(r,c)) * M2(r,c);
            return ret;
        }

        namespace detail {
            template<typename T> struct sv_type { typedef T type; };
            template<typename T>
            struct sv_type<std::complex<T> > { typedef T type; };
        }

        template<typename T, class MemoryBlock>
        void svd(matrix<T, MemoryBlock> M,
                 matrix<T, MemoryBlock> & U,
                 matrix<T, MemoryBlock>& V,
                 typename associated_real_diagonal_matrix<matrix<T, MemoryBlock> >::type& S)
        {
            BOOST_CONCEPT_ASSERT((alps::numeric::Matrix<matrix<T, MemoryBlock> >));
            typename matrix<T, MemoryBlock>::size_type k = std::min(num_rows(M), num_cols(M));
            resize(U, num_rows(M), k);
            resize(V, k, num_cols(M));
            resize(S, k, k);
            int info = boost::numeric::bindings::lapack::gesvd('S', 'S', M, S.get_values(), U, V);
            if (info != 0)
                throw std::runtime_error("Error in SVD!");
        }

        template<typename T, class MemoryBlock>
        void qr(matrix<T, MemoryBlock> M,
                matrix<T, MemoryBlock> & Q,
                matrix<T, MemoryBlock> & R)
        {
            typename matrix<T, MemoryBlock>::size_type k = std::min(num_rows(M), num_cols(M));
            typename associated_vector<matrix<typename detail::sv_type<T>::type, MemoryBlock> >::type tau(k);
            int info = 0; //boost::numeric::bindings::lapack::geqrf(M, tau);
            if (info != 0)
                throw std::runtime_error("Error in geqrf");

            resize(Q, num_rows(M), k);
            resize(R, k, num_cols(M));

            // get R
            std::fill(elements(R).first, elements(R).second, 0);
            for (std::size_t c = 0; c < num_cols(M); ++c)
                for (std::size_t r = 0; r <= c; ++r)
                    R(r, c) = M(r, c);

            // get Q from householder reflections in M
            std::fill(elements(Q).first, elements(Q).second, 0);
        }

        template<typename T, class MemoryBlock>
        matrix<T, MemoryBlock> exp (matrix<T, MemoryBlock> M, T const & alpha=1)
        {
            matrix<T, MemoryBlock> N, tmp;
            typename associated_real_vector<matrix<T, MemoryBlock> >::type Sv(num_rows(M));

            heev(M, N, Sv);

            typename associated_diagonal_matrix<matrix<T, MemoryBlock> >::type S(Sv);
            S = exp(alpha*S);
            gemm(N, S, tmp);
            gemm(tmp, adjoint(N), M);
            return M;
        }

        template<typename T, class MemoryBlock, class Generator>
        void generate(matrix<T, MemoryBlock>& m, Generator g)
        {
           std::generate(elements(m).first, elements(m).second, g);
        }

        template<typename T, class MemoryBlock>
        void heev(matrix<T, MemoryBlock> M
            , matrix<T, MemoryBlock> & evecs
            , typename associated_real_vector<matrix<T, MemoryBlock> >::type & evals
        )
        {
            assert(num_rows(M) == num_cols(M));
            assert(evals.size() == num_rows(M));
#ifndef NDEBUG
            for (int i = 0; i < num_rows(M); ++i)
                for (int j = 0; j < num_cols(M); ++j)
                    assert( abs( M(i,j) - conj(M(j,i)) ) < 1e-10 );
#endif
            boost::numeric::bindings::lapack::heevd('V', M, evals);
            // to be consistent with the SVD, I reorder in decreasing order
            std::reverse(evals.begin(), evals.end());
            // and the same with the matrix
            evecs.resize(num_rows(M), num_cols(M));
            for (std::size_t c = 0; c < num_cols(M); ++c)
                std::copy(col(M, c).first, col(M, c).second,
                          col(evecs, num_cols(M)-1-c).first);
        }

        template<typename T, class MemoryBlock>
        void heev(matrix<T, MemoryBlock> M,
                  matrix<T, MemoryBlock> & evecs,
                  typename associated_diagonal_matrix<matrix<T, MemoryBlock> >::type & evals)
        {
            assert(num_rows(M) == num_cols(M));
            typename associated_real_vector<matrix<T, MemoryBlock> >::type evals_(num_rows(M));
            heev(M, evecs, evals_);
            evals = typename associated_diagonal_matrix<matrix<T, MemoryBlock> >::type(evals_);
        }

        template<typename T, class MemoryBlock, class ThirdArgument>
        void syev(matrix<T, MemoryBlock> M,
                  matrix<T, MemoryBlock> & evecs,
                  ThirdArgument & evals)
        {
            heev(M, evecs, evals);
        }
        /*
        * Some block_matrix algorithms necessitate nested specialization due to ambient scheduler
        * the algos are full rewritten or partly with subset specialization 
        * an alternative implementation is presented inside p_dense_matrix/algorithms.hpp
        */
    } // end namspace numeric
} //end namespace alps

#endif //ALPS_MATRIX_ALGORITHMS_HPP
