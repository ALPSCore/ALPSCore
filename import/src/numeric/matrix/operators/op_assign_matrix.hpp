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
#ifndef ALPS_NUMERIC_OPERATORS_OP_ASSIGN_MATRIX_HPP
#define ALPS_NUMERIC_OPERATORS_OP_ASSIGN_MATRIX_HPP

#include <alps/numeric/matrix/entity.hpp>
#include <alps/numeric/matrix/is_blas_dispatchable.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/numeric/bindings/blas/level1/axpy.hpp>
#include <algorithm>
#include <functional>

namespace alps {
namespace numeric {

    namespace impl {
        template <typename Matrix1, typename Matrix2, typename Operation>
        void plus_minus_assign_impl(Matrix1& lhs, Matrix2 const& rhs, Operation op, tag::matrix, tag::matrix)
        {
            // One could do also a dispatch on row vs. column major, but since we don't have row major right now, let's leave it like that.
            typedef typename Matrix1::size_type             size_type;
            typedef typename Matrix1::col_element_iterator  col_element_iterator;
            typedef typename Matrix1::value_type            value_type;
            assert(num_rows(lhs) == num_rows(rhs));
            assert(num_cols(lhs) == num_cols(rhs));
#if defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
// Workaround for a compiler bug in clang 3.0 (and maybe earlier versions)
            for(size_type j=0; j < num_cols(lhs); ++j)
            {
                for(size_type i=0; i < num_rows(lhs); ++i)
                {
                    value_type const tmp = op(lhs(i,j),rhs(i,j));
                    lhs(i,j) = tmp;
                }
            }
#else //defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
            for(size_type j=0; j < num_cols(lhs); ++j)
            {
                std::pair<col_element_iterator,col_element_iterator> range(col(lhs,j));
                std::transform( range.first, range.second, col(rhs,j).first, range.first, op);
            }
#endif //defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
        }

        template <typename Matrix, typename T2>
        void multiplies_assign_impl(Matrix& lhs, T2 const& t, tag::matrix, tag::scalar)
        {
            typedef typename Matrix::size_type              size_type;
            typedef typename Matrix::value_type             value_type;
            typedef typename Matrix::col_element_iterator   col_element_iterator;
            // Do the operation column by column
            for(size_type j=0; j < num_cols(lhs); ++j)
            {
                std::pair<col_element_iterator,col_element_iterator> range(col(lhs,j));
                std::for_each(range.first, range.second, boost::lambda::_1 *= t);
            }
        }

    } // end namespace impl


    template <typename Matrix1, typename Matrix2>
    void plus_assign(Matrix1& lhs, Matrix2 const& rhs, tag::matrix tag1, tag::matrix tag2)
    {
        using impl::plus_minus_assign_impl;
        plus_minus_assign_impl(lhs,rhs,std::plus<typename Matrix1::value_type>(),tag1,tag2);
    }

    template <typename Matrix1, typename Matrix2>
    void minus_assign(Matrix1& lhs, Matrix2 const& rhs, tag::matrix tag1, tag::matrix tag2)
    {
        using impl::plus_minus_assign_impl;
        plus_minus_assign_impl(lhs,rhs,std::minus<typename Matrix1::value_type>(),tag1,tag2);
    }

    template <typename Matrix, typename T2>
    void multiplies_assign(Matrix& lhs, T2 const& t, tag::matrix tag1, tag::scalar tag2)
    {
        using impl::multiplies_assign_impl;
        multiplies_assign_impl(lhs,t,tag1,tag2);
    }

} // end namespace numeric
} // end namespace alps
#endif // ALPS_NUMERIC_OPERATORS_OP_ASSIGN_MATRIX_HPP
