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

#ifndef ALPS_MATRIX_CONJ_HPP
#define ALPS_MATRIX_CONJ_HPP

#include <alps/numeric/conj.hpp>

namespace alps {
namespace numeric {
/**
  * Does an conj_inplace on all elements of the matrix
  */
template <typename T, typename MemoryBlock>
void conj_inplace(matrix<T,MemoryBlock>& m)
{
    // TODO discuss conj() for matrix may be misleading:
    //      elementwise conj() <-> adjoint()
    //
    using std::for_each;
    typedef typename matrix<T,MemoryBlock>::col_element_iterator col_element_iterator;
    for(typename matrix<T,MemoryBlock>::size_type j=0; j < m.num_rows(); ++j)
       for(std::pair<col_element_iterator,col_element_iterator> range = m.col(j); range.first != range.second; ++range.first)
           conj_inplace(*range.first);
}

/**
  * Returns a matrix containing a the complex conjugates of the original matrix.
  * It does an element-wise conjugation.
  */
template <typename T, typename MemoryBlock>
matrix<T,MemoryBlock> conj(matrix<T,MemoryBlock> m)
{
    conj_inplace(m);
    return m;
}

} // end namespace numeric
} // end namespace alps 

#endif //ALPS_MATRIX_CONJ_HPP
