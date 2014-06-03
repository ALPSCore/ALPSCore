/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

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
