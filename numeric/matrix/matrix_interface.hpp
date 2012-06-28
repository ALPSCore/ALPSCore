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

#ifndef ALPS_MATRIX_INTERFACE_HPP
#define ALPS_MATRIX_INTERFACE_HPP

#include <alps/numeric/matrix/matrix_concept_check.hpp>

namespace alps {
    namespace numeric {
    // This macro creates free functions that call member functions with the same
    // name, e.g. swap_cols(A,i,j) -> A.swap_cols(i,j)
    #define COMMA ,
    #define IMPLEMENT_FORWARDING(TEMPLATE_PARS,TYPE,RET,NAME,ARGS,VARS) \
    template TEMPLATE_PARS \
    RET NAME ARGS \
    { \
        BOOST_CONCEPT_ASSERT((alps::numeric::Matrix<TYPE>)); \
        return m.NAME VARS; \
    }

    // num_rows(), num_cols(), swap_rows(), swap_cols()
    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         typename matrix<T COMMA MemoryBlock>::size_type, num_rows, (matrix<T, MemoryBlock> const& m), () )
    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         typename matrix<T COMMA MemoryBlock>::size_type, num_cols, (matrix<T, MemoryBlock> const& m), () )
    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         void, swap_rows, (matrix<T, MemoryBlock>& m, typename matrix<T, MemoryBlock>::size_type i1, typename matrix<T, MemoryBlock>::size_type i2), (i1,i2) )
    IMPLEMENT_FORWARDING((<typename T, class MemoryBlock>), matrix<T COMMA MemoryBlock>,
                         void, swap_cols, (matrix<T, MemoryBlock>& m, typename matrix<T, MemoryBlock>::size_type i1, typename matrix<T, MemoryBlock>::size_type i2), (i1,i2) ) 
    //
    // Matrix Iterator Interface
    // 
    #define ITERATOR_PAIR(TYPE, ITERATOR) \
    std::pair<typename TYPE::ITERATOR, typename TYPE::ITERATOR>

    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         ITERATOR_PAIR(matrix<T COMMA MemoryBlock>, row_element_iterator), row,
                         (matrix<T COMMA MemoryBlock> & m,
                          typename matrix<T COMMA MemoryBlock>::size_type i),
                         (i) )
    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         ITERATOR_PAIR(matrix<T COMMA MemoryBlock>, const_row_element_iterator), row,
                         (matrix<T COMMA MemoryBlock> const& m,
                          typename matrix<T COMMA MemoryBlock>::size_type i),
                         (i) )

    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         ITERATOR_PAIR(matrix<T COMMA MemoryBlock>, col_element_iterator), col,
                         (matrix<T COMMA MemoryBlock> & m,
                          typename matrix<T COMMA MemoryBlock>::size_type i),
                         (i) )
    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         ITERATOR_PAIR(matrix<T COMMA MemoryBlock>, const_col_element_iterator), col,
                         (matrix<T COMMA MemoryBlock> const& m,
                          typename matrix<T COMMA MemoryBlock>::size_type i),
                         (i) )

    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         ITERATOR_PAIR(matrix<T COMMA MemoryBlock>, element_iterator), elements,
                         (matrix<T COMMA MemoryBlock>& m), () )

    IMPLEMENT_FORWARDING(<typename T COMMA class MemoryBlock>, matrix<T COMMA MemoryBlock>,
                         ITERATOR_PAIR(matrix<T COMMA MemoryBlock>, const_element_iterator), elements,
                         (matrix<T COMMA MemoryBlock> const& m), () )
    #undef ITERATOR_PAIR
    #undef IMPLEMENT_FORWARDING
    #undef COMMA
    } //namespace numeric 
} //namespace alps     
#endif //ALPS_MATRIX_INTERFACE_HPP
