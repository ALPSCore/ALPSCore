/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2012 by Andreas Hehn <hehn@phys.ethz.ch>                   *
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

#ifndef ALPS_MATRIX_BLAS_HPP
#define ALPS_MATRIX_BLAS_HPP

#include <alps/numeric/matrix/detail/blasmacros.hpp>
#include <boost/numeric/bindings/blas/level3/gemm.hpp>
#include <boost/numeric/bindings/blas/level2/gemv.hpp>
#include <boost/numeric/bindings/blas/level1/axpy.hpp>
#include <boost/numeric/bindings/blas/level1/scal.hpp>

namespace alps {
    namespace numeric {
        template <typename T, typename MemoryBlock>
            class matrix;
        template <typename Matrix>
            class transpose_view;
    }
}


    //
    // matrix blas function hooks
    //
namespace alps {
    namespace numeric {
    #define ALPS_MATRIX_GEMM(T) \
        template <typename MemoryBlock> \
        void gemm( matrix<T,MemoryBlock> const& lhs, matrix<T,MemoryBlock> const& rhs, matrix<T,MemoryBlock> & result)\
        { \
            assert( !(lhs.num_cols() > rhs.num_rows()) ); \
            assert( !(lhs.num_cols() < rhs.num_rows()) ); \
            assert( lhs.num_cols() == rhs.num_rows() ); \
            assert( result.num_rows() == lhs.num_rows() ); \
            assert( result.num_cols() == rhs.num_cols() ); \
            boost::numeric::bindings::blas::gemm ( \
               typename matrix<T,MemoryBlock>::value_type(1), \
               lhs, \
               rhs, \
               typename matrix<T,MemoryBlock>::value_type(0), \
               result \
            ); \
        }\
        template <typename MemoryBlock> \
        void gemm(matrix<T,MemoryBlock> const& lhs, transpose_view<matrix<T,MemoryBlock> > const& rhs, matrix<T,MemoryBlock> & result) \
        { \
            assert( !(lhs.num_cols() > rhs.num_rows()) ); \
            assert( !(lhs.num_cols() < rhs.num_rows()) ); \
            assert( lhs.num_cols() == rhs.num_rows() ); \
            assert( result.num_rows() == lhs.num_rows() ); \
            assert( result.num_cols() == rhs.num_cols() ); \
            boost::numeric::bindings::blas::gemm( \
               typename matrix<T,MemoryBlock>::value_type(1), \
               lhs, \
               rhs, \
               typename matrix<T,MemoryBlock>::value_type(0), \
               result \
            ); \
        } \
        template <typename MemoryBlock> \
        void gemm(transpose_view<matrix<T,MemoryBlock> > const& lhs, matrix<T,MemoryBlock> const& rhs, matrix<T,MemoryBlock> & result) \
        { \
            assert( !(lhs.num_cols() > rhs.num_rows()) ); \
            assert( !(lhs.num_cols() < rhs.num_rows()) ); \
            assert( lhs.num_cols() == rhs.num_rows() ); \
            assert( result.num_rows() == lhs.num_rows() ); \
            assert( result.num_cols() == rhs.num_cols() ); \
            boost::numeric::bindings::blas::gemm( \
               typename matrix<T,MemoryBlock>::value_type(1), \
               lhs, \
               rhs, \
               typename matrix<T,MemoryBlock>::value_type(0), \
               result \
            ); \
        } \
        template <typename MemoryBlock> \
        void gemm(transpose_view<matrix<T,MemoryBlock> > const& lhs, transpose_view<matrix<T,MemoryBlock> > const& rhs, matrix<T,MemoryBlock> & result) \
        { \
            assert( !(lhs.num_cols() > rhs.num_rows()) ); \
            assert( !(lhs.num_cols() < rhs.num_rows()) ); \
            assert( lhs.num_cols() == rhs.num_rows() ); \
            assert( result.num_rows() == lhs.num_rows() ); \
            assert( result.num_cols() == rhs.num_cols() ); \
            boost::numeric::bindings::blas::gemm( \
               typename matrix<T,MemoryBlock>::value_type(1), \
               lhs, \
               rhs, \
               typename matrix<T,MemoryBlock>::value_type(0), \
               result \
            ); \
        }
    ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(ALPS_MATRIX_GEMM)
    #undef ALPS_MATRIX_GEMM

    #define MATRIX_VECTOR_MULTIPLY(T) \
        template <typename MemoryBlock, typename MemoryBlock2> \
        const vector<T,MemoryBlock2> matrix_vector_multiply(matrix<T,MemoryBlock> const& m, vector<T,MemoryBlock2> const& v) \
        { \
            assert( m.num_cols() == v.size()); \
            vector<T,MemoryBlock2> result(m.num_rows()); \
            boost::numeric::bindings::blas::gemv( \
                typename matrix<T,MemoryBlock>::value_type(1), \
                m, \
                v, \
                typename matrix<T,MemoryBlock>::value_type(0), \
                result \
            ); \
            return result; \
        }
    ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(MATRIX_VECTOR_MULTIPLY)
    #undef MATRIX_VECTOR_MULTIPLY

    // This seems to be the best solution for the *_ASSIGN dispatchers at the moment even though they call functions within the detail namespace
    #define PLUS_MINUS_ASSIGN(T) \
        template <typename MemoryBlock> \
        void plus_and_minus_assign_impl(matrix<T,MemoryBlock>& m, matrix<T,MemoryBlock> const& rhs, typename matrix<T,MemoryBlock>::value_type const& sign) \
        { \
            assert( m.num_cols() == rhs.num_cols() && m.num_rows() == rhs.num_rows() ); \
            if(!(m.is_shrinkable() || rhs.is_shrinkable()) ) \
            { \
                boost::numeric::bindings::blas::detail::axpy( m.num_rows() * m.num_cols(), sign, &(*rhs.col(0).first), 1, &(*m.col(0).first), 1); \
            } \
            else \
            { \
                for(std::size_t j=0; j < m.num_cols(); ++j) \
                    boost::numeric::bindings::blas::detail::axpy( m.num_rows(), sign, &(*rhs.col(j).first), 1, &(*m.col(j).first), 1); \
            } \
        } \
        template <typename MemoryBlock> \
        void plus_assign(matrix<T,MemoryBlock>& m, matrix<T,MemoryBlock> const& rhs) \
            { plus_and_minus_assign_impl(m, rhs, typename matrix<T,MemoryBlock>::value_type(1)); } \
        template <typename MemoryBlock> \
        void minus_assign(matrix<T,MemoryBlock>& m, matrix<T,MemoryBlock> const& rhs) \
            { plus_and_minus_assign_impl(m, rhs, typename matrix<T,MemoryBlock>::value_type(-1)); }
    ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(PLUS_MINUS_ASSIGN)
    #undef PLUS_MINUS_ASSIGN

    #define MULTIPLIES_ASSIGN(T) \
        template <typename MemoryBlock> \
        void multiplies_assign(matrix<T,MemoryBlock>& m, T const& t) \
        { \
            if( !(m.is_shrinkable()) ) \
            { \
                boost::numeric::bindings::blas::detail::scal( m.num_rows()*m.num_cols(), t, &(*m.col(0).first), 1 ); \
            } \
            else \
            { \
                for(std::size_t j=0; j <m.num_cols(); ++j) \
                    boost::numeric::bindings::blas::detail::scal( m.num_rows(), t, &(*m.col(j).first), 1 ); \
            } \
        }
        ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(MULTIPLIES_ASSIGN)
    #undef MULTIPLIES_ASSIGN

    } // end namespace numeric
} // end namespace alps

#endif // ALPS_MATRIX_BLAS_HPP
