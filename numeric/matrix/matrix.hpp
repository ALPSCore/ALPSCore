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

#ifndef ALPS_MATRIX_HPP
#define ALPS_MATRIX_HPP

#include <alps/numeric/matrix/strided_iterator.hpp>
#include <alps/numeric/matrix/matrix_element_iterator.hpp>
#include <alps/numeric/matrix/vector.hpp>
#include <alps/numeric/matrix/detail/matrix_adaptor.hpp>
#include <alps/numeric/matrix/matrix_traits.hpp>
#include <alps/numeric/real.hpp>

//#include "utils/function_objects.h"

#include <boost/lambda/lambda.hpp>
#include <boost/typeof/typeof.hpp>
#include <ostream>
#include <vector>
#include <algorithm>
#include <functional>
#include <cassert>

#ifdef HAVE_ALPS_HDF5
#include <alps/hdf5.hpp>
#include <boost/utility.hpp>
#include <alps/type_traits/is_complex.hpp>
#endif

namespace alps {
    namespace numeric {
    /** A matrix template class
      *
      * The dense_matrix class is a matrix which can take any type T
      * @param T the type of the elements to be stored in the matrix
      * @param MemoryBlock the underlying (continous) Memory structure
      */
    template <typename T, typename MemoryBlock = std::vector<T> >
    class dense_matrix {
    public:
        // typedefs required for a std::container concept
        typedef T                       value_type;       // The type T of the elements of the matrix
        typedef T&                      reference;        // Reference to value_type
        typedef T const&                const_reference;  // Const reference to value_type
        typedef std::size_t             size_type;        // Unsigned integer type that represents the dimensions of the matrix
        typedef std::ptrdiff_t          difference_type;  // Signed integer type to represent the distance of two elements in the memory

        // for compliance with an std::container one would also need
        // -operators == != < > <= >=
        // -size()
        // -typedefs iterator, const_iterator

        // typedefs for matrix specific iterators
        typedef strided_iterator<dense_matrix,value_type>
            row_element_iterator;                         // Iterator to iterate through the elements of a row of the matrix
        typedef strided_iterator<const dense_matrix,const value_type>
            const_row_element_iterator;                   // Const version of row_element_iterator
        typedef value_type*
            column_element_iterator;                      // Iterator to iterate through the elements of a columns of the matrix
        typedef value_type const*
            const_column_element_iterator;                // Const version of column_element_iterator       
        typedef matrix_element_iterator<dense_matrix,value_type>
            element_iterator;                             // Iterator to iterate through all elements of the matrix (REALLY SLOW! USE row_-/column_iterators INSTEAD!)
        typedef matrix_element_iterator<const dense_matrix,const value_type>
            const_element_iterator;                       // Const version of element_iterator (REALLY SLOW! USE row_-/column_iterators INSTEAD!)

        /**
          * Static function for creating identiy matrix
          *
          */
        static dense_matrix<T,MemoryBlock> identity_matrix(size_type size);

        /**
          * The constructor
          * @param rows the number of rows
          * @param columns the number of columns
          * @param init_value all matrix elements will be initialized to this value.
          */
        explicit dense_matrix(size_type rows = 0, size_type cols = 0, T init_value = T());

        /**
          * The copy constructor
          *
          */
        template <typename OtherMemoryBlock>
        dense_matrix(dense_matrix<T,OtherMemoryBlock> const& m);

        /**
          * Non-throwing swap function
          * @param r dense_matrix object which should be swapped with the dense_matrix (this)
          */
        void swap(dense_matrix & r);

        /**
          * Swaps two dense_matrices
          */
        friend void swap(dense_matrix & x, dense_matrix & y)
        {
            x.swap(y);
        }

        /**
          * Assigns the matrix to matrix rhs
          */
        dense_matrix& operator = (dense_matrix rhs);

        /**
          * Access the element in row i, column j
          * @param i 0<= i <= num_rows()
          * @param j 0<= j <= num_cols()
          * @return A mutable reference to the matrix element at position (i,j).
          */
        inline value_type& operator()(const size_type i, const size_type j);
        
        /**
          * Access the element in row i, column j
          * @return A constant reference to the matrix element at position (i,j).
          */
        inline value_type const& operator()(const size_type i, const size_type j) const;

        bool operator == (dense_matrix const& rhs) const;

        dense_matrix<T,MemoryBlock>& operator += (dense_matrix const& rhs); 

        dense_matrix<T,MemoryBlock>& operator -= (dense_matrix const& rhs);
        
        template <typename T2>
        dense_matrix<T,MemoryBlock>& operator *= (T2 const& t);
        
        template <typename T2>
        dense_matrix<T,MemoryBlock>& operator /= (T2 const& t);

        /**
          * Checks if a matrix is empty
          * @return true if the matrix is a 0x0 matrix, false otherwise.
          */
        inline bool empty() const;

        /**
          * @return number of rows of the matrix
          */
        inline size_type num_rows() const;

        /**
          * @return number of columns of the matrix
          */
        inline size_type num_cols() const;

        /**
          * @return The stride for moving to the next element along a row
          */
        inline difference_type stride1() const;
        
        /**
          * @return The stride for moving to the next element along a column
          */
        inline difference_type stride2() const;

        /** Resize the matrix.
          * Resizes the matrix to size1 rows and size2 columns and initializes
          * the new elements to init_value. It also enlarges the MemoryBlock
          * if needed. If the new size for any dimension is
          * smaller only elements outside the new size will be deleted.
          * If the new size is larger for any dimension the new elements
          * will be initialized by the init_value.
          * All other elements will keep their value.  
          *
          * Exception behaviour:
          * As long as the assignment and copy operation of the T values don't throw an exception,
          * any exception will leave the matrix unchanged.
          * (Assuming the same behaviour of the underlying MemoryBlock. This is true for std::vector.)
          * @param size1 new number of rows
          * @param size2 new number of columns
          * @param init_value value to which the new elements will be initalized
          */
        void resize(size_type size1, size_type size2, T const & init_value = T());
        
        /**
          * Reserves memory for anticipated enlargements of the matrix
          * @param size1 For how many rows should memory be reserved, value is ignored if it's smaller than the current number of rows
          * @param size2 For how many columns should memory be reserved, value is ignored if it's smaller than the current number of columns
          * @param init_value i
          */
        void reserve(size_type size1, size_type size2, T const & init_value = T());

        std::pair<size_type,size_type> capacity() const;
        
        bool is_shrinkable() const;

        void clear();

        std::pair<row_element_iterator,row_element_iterator> row(size_type row = 0)
        {
            return std::make_pair( row_element_iterator(&values_[row],reserved_size1_), row_element_iterator(&values_[row+reserved_size1_*size2_], reserved_size1_) );
        }

        std::pair<const_row_element_iterator,const_row_element_iterator> row(size_type row = 0) const
        {
            return std::make_pair( const_row_element_iterator(&values_[row],reserved_size1_), const_row_element_iterator(&values_[row+reserved_size1_*size2_], reserved_size1_) );
        }
        std::pair<column_element_iterator,column_element_iterator> column(size_type col = 0 )
        {
            return std::make_pair( column_element_iterator(&values_[col*reserved_size1_]), column_element_iterator(&values_[col*reserved_size1_+size1_]) );
        }
        std::pair<const_column_element_iterator,const_column_element_iterator> column(size_type col = 0 ) const
        {
            return std::make_pair( const_column_element_iterator(&values_[col*reserved_size1_]), const_column_element_iterator(&values_[col*reserved_size1_+size1_]) );
        }
        std::pair<element_iterator,element_iterator> elements()
        {
            return std::make_pair( element_iterator(this,0,0), element_iterator(this,0, num_cols()) );
        }
        std::pair<element_iterator,element_iterator> elements() const
        {
            return std::make_pair( const_element_iterator(this,0,0), const_element_iterator(this,0,num_cols() ) );
        }

        template <typename InputIterator>
        void append_cols(std::pair<InputIterator,InputIterator> const& range, difference_type k = 1);

        template <typename InputIterator>
        void append_rows(std::pair<InputIterator,InputIterator> const& range, difference_type k = 1);

        template <typename InputIterator>
        void insert_rows(size_type i, std::pair<InputIterator,InputIterator> const& range, difference_type k = 1);

        template <typename InputIterator>
        void insert_cols(size_type j, std::pair<InputIterator,InputIterator> const& range, difference_type k = 1);

        void remove_rows(size_type i, difference_type k = 1);

        void remove_cols(size_type j, difference_type k = 1);

        void swap_rows(size_type i1, size_type i2);

        void swap_cols(size_type j1, size_type j2);

        void inplace_conjugate();
		
		// Serialize functions to save dense_matrix with alps::hdf5
#ifdef HAVE_ALPS_HDF5
        void load_impl(alps::hdf5::archive & ar, boost::mpl::true_);
        void load_impl(alps::hdf5::archive & ar, boost::mpl::false_);

		void load(alps::hdf5::archive & ar);
		void save(alps::hdf5::archive & ar) const;
#endif
        MemoryBlock const& get_values() const;       
        MemoryBlock & get_values();       


    private:
        template <typename OtherT,typename OtherMemoryBlock>
        friend class dense_matrix;


        inline bool automatic_reserve(size_type size1, size_type size2, T const& init_value = T());

        size_type size1_;
        size_type size2_;
        size_type reserved_size1_;
        // "reserved_size2_" is done automatically by underlying std::vector (see vector.reserve(), vector.capacity() )
        MemoryBlock values_;
    };
    
    }  // namespace numeric 
} // namespace alps 

//
// Function hooks
//
namespace alps {
    namespace numeric { 

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> matrix_matrix_multiply(dense_matrix<T,MemoryBlock> const& lhs, dense_matrix<T,MemoryBlock> const& rhs);
    
    template<typename T, typename MemoryBlock, typename T2, typename MemoryBlock2>
    typename matrix_vector_multiplies_return_type<dense_matrix<T,MemoryBlock>,vector<T2,MemoryBlock2> >::type
    matrix_vector_multiply(dense_matrix<T,MemoryBlock> const& m, vector<T2,MemoryBlock2> const& v);
    
    template <typename T,typename MemoryBlock>
    void plus_assign(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs);

    template <typename T, typename MemoryBlock>
    void minus_assign(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs);

    template <typename T, typename MemoryBlock, typename T2>
    void multiplies_assign(dense_matrix<T,MemoryBlock>& m, T2 const& t);

    }
}

//
// Free dense matrix functions
//
namespace alps {
    namespace numeric {

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator + (dense_matrix<T,MemoryBlock> a, dense_matrix<T,MemoryBlock> const& b);
    
    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator - (dense_matrix<T,MemoryBlock> a, dense_matrix<T,MemoryBlock> const& b);

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator - (dense_matrix<T,MemoryBlock> a);

    template<typename T, typename MemoryBlock, typename T2, typename MemoryBlock2>
    typename matrix_vector_multiplies_return_type<dense_matrix<T,MemoryBlock>,vector<T2,MemoryBlock2> >::type
    operator * (dense_matrix<T,MemoryBlock> const& m, vector<T2,MemoryBlock2> const& v);
   
    // TODO: adj(Vector) * Matrix, where adj is a proxy object

    template<typename T,typename MemoryBlock, typename T2>
    const dense_matrix<T,MemoryBlock> operator * (dense_matrix<T,MemoryBlock> m, T2 const& t);
    
    template<typename T,typename MemoryBlock, typename T2>
    const dense_matrix<T,MemoryBlock> operator * (T2 const& t, dense_matrix<T,MemoryBlock> m);

    template<typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator * (dense_matrix<T,MemoryBlock> const& m1, dense_matrix<T,MemoryBlock> const& m2);

    template<typename T,typename MemoryBlock>
    void gemm(dense_matrix<T,MemoryBlock> const & A, dense_matrix<T,MemoryBlock> const & B, dense_matrix<T,MemoryBlock> & C);
    
    template<class T, class MemoryBlock>
    std::size_t size_of(dense_matrix<T, MemoryBlock> const & m);
    
    template <typename T, typename MemoryBlock>
    std::ostream& operator << (std::ostream& o, dense_matrix<T,MemoryBlock> const& m);

    } // namespace numeric
} // namespace alps


//
// Trait specializations
//
namespace alps {
    namespace numeric {

    //
    // Forward declarations
    //
    template <typename T>
    class diagonal_matrix;
    
    template <typename T, typename MemoryBlock>
    class vector;

    template<typename T, typename MemoryBlock>
    struct associated_diagonal_matrix<dense_matrix<T, MemoryBlock> >
    {
        typedef  diagonal_matrix<T> type;
    };
    
    template<typename T, typename MemoryBlock>
    struct associated_real_diagonal_matrix<dense_matrix<T, MemoryBlock> >
    {
        typedef diagonal_matrix<typename real_type<T>::type> type;
    };
    
    template <typename T1, typename MemoryBlock1, typename T2, typename MemoryBlock2>
    struct matrix_vector_multiplies_return_type<dense_matrix<T1,MemoryBlock1>,vector<T2,MemoryBlock2> >
    {
        private:
            typedef char one;
            typedef long unsigned int two;
            static one test(T1 t) {return one();}
            static two test(T2 t) {return two();}
            typedef typename boost::mpl::if_<typename boost::mpl::bool_<(sizeof(test(T1()*T2())) == sizeof(one))>,T1,T2>::type value_type;
            typedef typename boost::mpl::if_<typename boost::mpl::bool_<(sizeof(test(T1()*T2())) == sizeof(one))>,MemoryBlock1,MemoryBlock2>::type memoryblock_type;
        public:
            typedef alps::numeric::vector<value_type,memoryblock_type> type;
    };

    template <typename T,typename MemoryBlock1, typename MemoryBlock2>
    struct matrix_vector_multiplies_return_type<dense_matrix<T,MemoryBlock1>,vector<T,MemoryBlock2> >
    {
        typedef alps::numeric::vector<T,MemoryBlock2> type;
    };

    }
}

#include <alps/numeric/matrix/matrix.ipp>
#endif //ALPS_MATRIX_HPP
