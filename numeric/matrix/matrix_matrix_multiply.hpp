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


namespace alps {
namespace numeric {
    //
    // Default matrix matrix multiplication implementations
    // May be overlaoded for special Matrix types
    //

    // The classic gemm - as known from Fortran - writing the result to argument c
    template<typename MatrixA, typename MatrixB, typename MatrixC>
    void gemm(MatrixA const & a, MatrixB const & b, MatrixC & c)
    {
        assert( num_cols(a) == num_rows(b) );
        assert( num_rows(c) == num_rows(a) );
        assert( num_cols(c) == num_cols(b) );
        // Simple matrix matrix multiplication
        std::cerr << "Warning: unoptimized version of GEMM is being used!\n";
        for(std::size_t j=0; j < num_cols(b); ++j)
            for(std::size_t k=0; k < num_cols(a); ++k)
                for(std::size_t i=0; i < num_rows(a); ++i)
                    c(i,j) += a(i,k) * b(k,j);
    }

    //
    // A multiply function returning the resulting matrix
    //
    // In most cases you should prefer this function to the gemm().
    //
    template <typename Matrix1, typename Matrix2>
    typename matrix_matrix_multiply_return_type<Matrix1,Matrix2>::type matrix_matrix_multiply(Matrix1 const& lhs, Matrix2 const& rhs)
    {
        typename matrix_matrix_multiply_return_type<Matrix1,Matrix2>::type result(lhs.num_rows(),rhs.num_cols());
        gemm(lhs,rhs,result);
        return result;
    }
} // end namespace numeric
} // end namespace alps
