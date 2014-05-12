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
#ifndef ALPS_NUMERIC_MATRIX_OPERATORS_MULTIPLY_MATRIX_HPP
#define ALPS_NUMERIC_MATRIX_OPERATORS_MULTIPLY_MATRIX_HPP

#include <alps/numeric/matrix/gemm.hpp>
#include <alps/numeric/matrix/gemv.hpp>

namespace alps {
namespace numeric {

template <typename Matrix1, typename Matrix2>
typename multiply_return_type_helper<Matrix1,Matrix2>::type multiply(Matrix1 const& m1, Matrix2 const& m2, tag::matrix, tag::matrix)
{
    typename multiply_return_type_helper<Matrix1,Matrix2>::type r(num_rows(m1),num_cols(m2));
    gemm(m1,m2,r);
    return r;
}

template <typename Matrix, typename Vector>
typename multiply_return_type_helper<Matrix,Vector>::type multiply(Matrix const& m, Vector const& v, tag::matrix, tag::vector)
{
    typename multiply_return_type_helper<Matrix,Vector>::type r(num_rows(m));
    gemv(m,v,r);
    return r;
}

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_OPERATORS_MULTIPLY_MATRIX_HPP
