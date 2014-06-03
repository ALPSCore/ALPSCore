/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
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
