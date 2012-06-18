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
    
namespace alps {
    namespace numeric {
        template <typename T, typename MemoryBlock>
            class dense_matrix;
    }
}    


    //
    // dense matrix blas function hooks
    //
namespace alps {
    namespace numeric {
    #define MATRIX_MATRIX_MULTIPLY(T) \
        template <typename MemoryBlock> \
        const dense_matrix<T,MemoryBlock> matrix_matrix_multiply(dense_matrix<T,MemoryBlock> const& lhs, dense_matrix<T,MemoryBlock> const& rhs) \
        { \
            assert( !(lhs.num_cols() > rhs.num_rows()) ); \
            assert( !(lhs.num_cols() < rhs.num_rows()) ); \
            assert( lhs.num_cols() == rhs.num_rows() ); \
            dense_matrix<T,MemoryBlock> result(lhs.num_rows(),rhs.num_cols()); \
            boost::numeric::bindings::blas::gemm \
                ( \
                   typename dense_matrix<T,MemoryBlock>::value_type(1), \
                   lhs, \
                   rhs, \
                   typename dense_matrix<T,MemoryBlock>::value_type(0), \
                   result \
                ); \
            return result; \
        }
    ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(MATRIX_MATRIX_MULTIPLY)
    #undef MATRIX_MATRIX_MULTIPLY
    
    #define MATRIX_VECTOR_MULTIPLY(T) \
        template <typename MemoryBlock, typename MemoryBlock2> \
        const vector<T,MemoryBlock2> matrix_vector_multiply(dense_matrix<T,MemoryBlock> const& m, vector<T,MemoryBlock2> const& v) \
        { \
            assert( m.num_cols() == v.size()); \
            vector<T,MemoryBlock2> result(m.num_rows()); \
            boost::numeric::bindings::blas::gemv \
                ( \
                  typename dense_matrix<T,MemoryBlock>::value_type(1), \
                  m, \
                  v, \
                  typename dense_matrix<T,MemoryBlock>::value_type(0), \
                  result \
                ); \
            return result; \
        }
    ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(MATRIX_VECTOR_MULTIPLY)
    #undef MATRIX_VECTOR_MULTIPLY
    
    // This seems to be the best solution for the *_ASSIGN dispatchers at the moment even though they call functions within the detail namespace
    #define PLUS_MINUS_ASSIGN(T) \
        template <typename MemoryBlock> \
        void plus_and_minus_assign_impl(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs, typename dense_matrix<T,MemoryBlock>::value_type const& sign) \
        { \
            assert( m.num_cols() == rhs.num_cols() && m.num_rows() == rhs.num_rows() ); \
            if(!(m.is_shrinkable() || rhs.is_shrinkable()) ) \
            { \
                boost::numeric::bindings::blas::detail::axpy( m.num_rows() * m.num_cols(), sign, &(*rhs.column(0).first), 1, &(*m.column(0).first), 1); \
            } \
            else \
            { \
                for(std::size_t j=0; j < m.num_cols(); ++j) \
                    boost::numeric::bindings::blas::detail::axpy( m.num_rows(), sign, &(*rhs.column(j).first), 1, &(*m.column(j).first), 1); \
            } \
        } \
        template <typename MemoryBlock> \
        void plus_assign(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs) \
            { plus_and_minus_assign_impl(m, rhs, typename dense_matrix<T,MemoryBlock>::value_type(1)); } \
        template <typename MemoryBlock> \
        void minus_assign(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs) \
            { plus_and_minus_assign_impl(m, rhs, typename dense_matrix<T,MemoryBlock>::value_type(-1)); }
    ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(PLUS_MINUS_ASSIGN)
    #undef PLUS_MINUS_ASSIGN
    
    #define MULTIPLIES_ASSIGN(T) \
        template <typename MemoryBlock> \
        void multiplies_assign(dense_matrix<T,MemoryBlock>& m, T const& t) \
        { \
            if( !(m.is_shrinkable()) ) \
            { \
                boost::numeric::bindings::blas::detail::scal( m.num_rows()*m.num_cols(), t, &(*m.column(0).first), 1 ); \
            } \
            else \
            { \
                for(std::size_t j=0; j <m.num_cols(); ++j) \
                    boost::numeric::bindings::blas::detail::scal( m.num_rows(), t, &(*m.column(j).first), 1 ); \
            } \
        }
        ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(MULTIPLIES_ASSIGN)
    #undef MULTIPLIES_ASSIGN
    
    } // end namespace numeric
} // end namespace alps

#endif // ALPS_MATRIX_BLAS_HPP
