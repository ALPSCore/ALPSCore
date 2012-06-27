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

#ifndef __ALPS_RESIZABLE_MATRIX_INTERFACE_HPP__
#define __ALPS_RESIZABLE_MATRIX_INTERFACE_HPP__

#include <alps/numeric/matrix/resizable_matrix_concept_check.hpp>

namespace alps {
    namespace numeric { 
    // resize(), remove_rows(), remove_cols()
    template <typename T, typename MemoryBlock>
    void resize(matrix<T,MemoryBlock>& m, typename matrix<T,MemoryBlock>::size_type i, typename matrix<T,MemoryBlock>::size_type j)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.resize(i,j);
    }
    
    template <typename T, typename MemoryBlock>
    void resize( matrix<T,MemoryBlock>& m,
            typename matrix<T,MemoryBlock>::size_type i,
            typename matrix<T,MemoryBlock>::size_type j,
            typename matrix<T,MemoryBlock>::value_type const& t )
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.resize(i,j,t);
    }
    
    template <typename T, typename MemoryBlock>
    void remove_rows( matrix<T,MemoryBlock>& m,
            typename matrix<T,MemoryBlock>::size_type i,
            typename matrix<T,MemoryBlock>::difference_type k = 1)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.remove_rows(i,k);
    }
    
    template <typename T, typename MemoryBlock>
    void remove_cols( matrix<T,MemoryBlock>& m,
            typename matrix<T,MemoryBlock>::size_type j,
            typename matrix<T,MemoryBlock>::difference_type k = 1)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.remove_cols(j,k);
    }
    
    //append_rows(), append_cols(), insert_rows(), insert_cols()
    #define INPUT_ITERATOR_PAIR std::pair<InputIterator,InputIterator>
    
    template <typename T, typename MemoryBlock, typename InputIterator>
    void append_rows( matrix<T,MemoryBlock>& m, INPUT_ITERATOR_PAIR range,
            typename matrix<T,MemoryBlock>::difference_type k = 1)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.append_rows(range,k);
    }
    
    template <typename T, typename MemoryBlock, typename InputIterator>
    void append_cols( matrix<T,MemoryBlock>& m, INPUT_ITERATOR_PAIR range,
            typename matrix<T,MemoryBlock>::difference_type k = 1)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.append_cols(range,k);
    }
    
    template <typename T, typename MemoryBlock, typename InputIterator>
    void insert_rows( matrix<T,MemoryBlock>& m,
            typename matrix<T,MemoryBlock>::size_type i,
            INPUT_ITERATOR_PAIR range,
            typename matrix<T,MemoryBlock>::difference_type k = 1)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.insert_rows(i,range,k);
    }
    
    template <typename T, typename MemoryBlock, typename InputIterator>
    void insert_cols( matrix<T,MemoryBlock>& m,
            typename matrix<T,MemoryBlock>::size_type j,
            INPUT_ITERATOR_PAIR range,
            typename matrix<T,MemoryBlock>::difference_type k = 1)
    {
        BOOST_CONCEPT_ASSERT((alps::numeric::ResizableMatrix<matrix<T,MemoryBlock> >));
        return m.insert_cols(j,range,k);
    }
    
    #undef INPUT_ITERATOR_PAIR
    
    } // namespace numeric
} // namespace alps
#endif //__ALPS_RESIZABLE_MATRIX_INTERFACE_HPP__
