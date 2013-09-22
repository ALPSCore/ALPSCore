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
#ifndef ALPS_NUMERIC_MATRIX_DETAIL_PRINT_MATRIX_HPP
#define ALPS_NUMERIC_MATRIX_DETAIL_PRINT_MATRIX_HPP

#include <ostream>


namespace alps {
namespace numeric {
namespace detail {

template <typename Matrix>
void print_matrix(std::ostream& os, Matrix const& m)
{
    os << "[";
    for(typename Matrix::size_type i=0; i < num_rows(m); ++i)
    {
        os << "[ ";
        if(num_cols(m) > 0)
        {
            for(typename Matrix::size_type j=0; j < num_cols(m)-1; ++j)
                os << m(i,j) << ", ";
            os << m(i,num_cols(m)-1);
        }
        os << "]";
        if(i+1 < num_rows(m))
            os << "," << std::endl;
    }
    os << "]" << std::endl;
}

} // end namespace detail
} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_DETAIL_PRINT_MATRIX_HPP
