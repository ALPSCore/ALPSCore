/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2013 by Andreas Hehn <hehn@phys.ethz.ch>                   *
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
#ifndef ALPS_NUMERIC_MATRIX_GEMV_HPP
#define ALPS_NUMERIC_MATRIX_GEMV_HPP

#include <alps/numeric/matrix/detail/debug_output.hpp>
#include <alps/numeric/matrix/is_blas_dispatchable.hpp>
#include <boost/numeric/bindings/blas/level2/gemv.hpp>
#include <cassert>

namespace alps {
namespace numeric {

template <typename Matrix, typename Vector, typename Vector2>
void gemv(Matrix const& m, Vector const& x, Vector2& y, boost::mpl::false_)
{
    typedef typename Matrix::size_type size_type;
    assert(num_cols(m) > 0);
    for(size_type j=0; j < num_cols(m); ++j)
    for(size_type i=0; i < num_rows(m); ++i)
        y[i] += m(i,j) * x[j];
}

template <typename Matrix, typename Vector, typename Vector2>
void gemv(Matrix const& m, Vector const& x, Vector2& y, boost::mpl::true_)
{
    ALPS_NUMERIC_MATRIX_DEBUG_OUTPUT( "using blas gemv for " << typeid(m).name() << " " << typeid(x).name() << " -> " << typeid(y).name() );
    typedef typename Matrix::value_type value_type;
    boost::numeric::bindings::blas::gemv(value_type(1), m, x, value_type(0), y);
}

template <typename Matrix, typename Vector, typename Vector2>
void gemv(Matrix const& m, Vector const& x, Vector2& y)
{
    assert(num_rows(m) == y.size());
    assert(num_cols(m) == x.size());
    // TODO test also Vector2
    gemv(m,x,y,is_blas_dispatchable<Matrix,Vector>());
}

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_GEMV_HPP
