#ifndef __ALPS_GENERAL_MATRIX_HPP__
#define __ALPS_GENERAL_MATRIX_HPP__

#include "blasmacros.h"
#include "matrix_iterators.hpp"
#include "../vector.hpp"

#include <ostream>
#include <vector>
#include <algorithm>
#include <functional>

#include <boost/numeric/bindings/detail/adaptor.hpp>
#include <boost/numeric/bindings/detail/if_row_major.hpp>
#include <boost/numeric/bindings/lapack/driver/gesdd.hpp>

//
// general_matrix template class
//
namespace blas {

    template <typename T>
    class general_matrix {
    public:
        // typedefs required for a std::container concept
        typedef T                       value_type;
        typedef T&                      reference;
        typedef T const&                const_reference;
        typedef std::size_t             size_type;
        typedef size_type               difference_type;

        // TODO
        // for compliance with an std::container one would alos need
        // -operators == != < > <= >=
        // -size()
        // -typedefs iterator, const_iterator
        // is it useful to implement this?

        // typedefs for matrix specific iterators
        // row_iterator: iterates over the rows of a specific column
        typedef matrix_row_iterator<general_matrix,value_type>                  row_iterator;
        typedef matrix_row_iterator<const general_matrix,const value_type>      const_row_iterator;
        // column_iterator: iterates over the columns of a specific row
        typedef matrix_column_iterator<general_matrix,value_type>               column_iterator;
        typedef matrix_column_iterator<const general_matrix,const value_type>   const_column_iterator;





        //TODO: alignment!
        general_matrix(std::size_t size1 = 0, std::size_t size2 = 0, T init_value = T(0) )
        : size1_(size1), size2_(size2), reserved_size1_(size1), values_(size1*size2, init_value)
        {
        }

        general_matrix(general_matrix<T> const& mat)
        : size1_(mat.size1_), size2_(mat.size2_), reserved_size1_(mat.size1_), values_(mat.size1_*mat.size2_)
        {
            // If the size of the matrix corresponds to the allocated size of the matrix...
            if(!is_shrinkable())
            {
                std::copy( mat.values_.begin(), mat.values_.end(), values_.begin() );
            }
            else
            {
                // copy only a shrinked to size version of the original matrix
                for(std::size_t j=0; j < mat.size2_; ++j)
                    std::copy( mat.rows_begin(j), mat.rows_end(j), this->rows_begin(j) );
            }
        }

        void swap(general_matrix<T> & r)
        {
            std::swap(values_, r.values_);
            std::swap(size1_, r.size1_);
            std::swap(size2_, r.size2_);
            std::swap(reserved_size1_,r.reserved_size1_);
        }
        
        friend void swap(general_matrix<T> & x, general_matrix<T> & y)
        {
            x.swap(y);
        }
        
        general_matrix& operator = (general_matrix<T> rhs)
        {
            // swap(rhs, *this); // anyone have any idea why this doesn't work?
            this->swap(rhs);
            return *this;
        }

        /*
           // TODO: Can these functions be removed?
           // This would hide the implementation from the user
        inline const std::vector<T> values() const
        {
            return values_; 
        }
        
        inline std::vector<T> &values() 
        {
            return values_; 
        }
        */

        inline T &operator()(const std::size_t i, const std::size_t j)
        {
            assert((i < size1_) && (j < size2_));
            return values_[i+j*reserved_size1_];
        }
        
        inline const T &operator()(const std::size_t i, const std::size_t j) const 
        {
            assert((i < size1_) && (j < size2_));
            return values_[i+j*reserved_size1_];
        }

        inline const bool empty() const
        {
            return (size1_ == 0 || size2_ == 0);
        }

        inline const std::size_t num_rows() const
        {
            return size1_;
        }

        inline const std::size_t num_columns() const
        {
            return size2_;
        }
        
        //TODO: shall these two functions be kept for compatibility or can we drop them?
        inline const std::size_t size1() const 
        {
            return size1_;
        }
  
        inline const std::size_t size2() const
        { 
            return size2_;
        }
       
        inline void resize(std::size_t size1, std::size_t size2, T init_value = T())
        {
           // Resizes the matrix to the size1 and size2 and allocates enlarges the vector if needed
           // If the new size for any dimension is smaller only elements outside the new size will be deleted.
           // If the new size is larger for any dimension the new elements will be initialized by the init_value.
           // All other elements will keep their value.

            // TODO: Over-resize matrix to 1.4 or 2 times the requested size
            if(size1 <= reserved_size1_)
            {
                   //TODO Exception safe? -> Are resize() and fill() exception safe?
                    values_.resize(reserved_size1_*size2_,init_value);

                if(size1 > size1_)
                {
                    // Reset all new elements which are in already reserved rows of already existing columns to init_value
                    // For all elements of new columns this is already done by values_.resize()
                    for(std::size_t j=0; j < size2_; ++j)
                    {
                        std::fill(values_.begin()+j*reserved_size1_ + size1_, values_.begin()+j*reserved_size1_ + size1, init_value);
                    }
                }

            }
            else // size1 > reserved_size1_
            {
                std::vector<T> tmp(size1*size2,init_value);
                for(std::size_t j=0; j < size2_; ++j)
                {
                    // Copy column by column
                    std::copy( values_.begin()+j*reserved_size1_, values_.begin()+j*reserved_size1_+size1_, tmp.begin()+j*size1);
                }
                std::swap(values_,tmp);
                reserved_size1_ = size1;
            }
            size1_=size1;
            size2_=size2;
        }
        
        inline void reserve(std::size_t size1, std::size_t size2)
        {
            if(size1*size2 > values_.capacity() )
            {
                    values_.reserve(size1*size2);
            }
            if(size1 > reserved_size1_)
            {
                std::vector<T> tmp(size1*size2);
                for(std::size_t j=0; j < size2_; ++j)
                {
                    // Copy column by column
                    std::copy( values_.begin()+j*reserved_size1_, values_.begin()+j*reserved_size1_+size1_, tmp.begin()+j*size1);
                }
                std::swap(values_,tmp);
                reserved_size1_ = size1;
            }
        }

        std::pair<std::size_t,std::size_t> capacity() const
        {
            assert( values_.capacity() % reserved_size1_ == 0 );
            // Evaluate the maximal number of columns (with size reserved_size1_) that the underlying vector could hold.
            // If the constructor, resize() and reserve() of std::vector would guarantee to allocate 
            // the requested amount of memory exactly
            // values_.capacity() % reserved_size1_ == 0 should hold.
            // However these functions guarantee to allocate _at least_ the requested amount.
            std::size_t reserved_size2_ = values_.capacity() - (values_.capacity() % reserved_size1_) / reserved_size1_;
            return std::pair<std::size_t,std::size_t>( reserved_size1_, reserved_size2_ );
        }
        
        bool is_shrinkable() const
        {
            // This assertion should actually never fail
            assert( reserved_size1_*size2_ == values_.size() );
            if(size1_ < reserved_size1_) return true;
            else return false;
        }

        void clear()
        {
            // Clear the values vector and ensure the reserved size stays the way it was
            values_.clear();
            values_.resize(reserved_size1_*size2_);
            size1_ = 0;
            size2_ = 0;
        }

        row_iterator rows_begin(std::size_t column=0)
        {
            assert( column < size2_);
            return row_iterator(this,0,column);
        }

        const_row_iterator rows_begin(std::size_t column=0) const
        {
            assert( column < size2_);
            return const_row_iterator(this,0,column);
        }

        row_iterator rows_end(std::size_t column=0)
        {
            assert( column < size2_);
            return row_iterator(this,size1_,column);
        }
        
        const_row_iterator rows_end(std::size_t column=0) const
        {
            assert( column < size2_);
            return const_row_iterator(this,size1_,column);
        }
        
        column_iterator columns_begin(std::size_t row=0)
        {
            assert( row < size1_ );
            return column_iterator(this,row,0);
        }

        const_column_iterator columns_begin(std::size_t row=0) const
        {
            assert( row < size1_ );
            return const_column_iterator(this,row,0);
        }

        column_iterator columns_end(std::size_t row=0)
        {
            assert( row < size1_ );
            return column_iterator(this,row,size2_);
        }
        
        const_column_iterator columns_end(std::size_t row=0) const
        {
            assert( row < size1_ );
            return const_column_iterator(this,row,size2_);
        }

        void append_column(vector<T> const& v)
        {
            assert( v.size() == size1_ );
            std::size_t insert_position = size2_;
            resize(size1_,size2_+1);    // This call modifies size2_ !
            std::copy( v.begin(), v.end(), rows_begin(insert_position) );
        }

        void apped_row(vector<T> const& v)
        {
            assert( v.size() == size2_ );
            std::size_t insert_position = size1_;
            resize(size1_+1,size2_);   // This call modifies size1_ !
            std::copy( v.begin(), v.end(), columns_begin(insert_position) );
        }

        void insert_row(std::size_t i, vector<T> const& v)
        {
            assert( i <= size1_ );
            assert( v.size() == size2_ );

            // Append the row
            append_row(v);

            // Move the row through the matrix to the right possition
            for(std::size_t k=size1_-1; k>i; ++k)
            {
                swap_rows(k,k-1);
            }
        }

        void insert_column(std::size_t j, vector<T> const& v)
        {
            assert( j <= size2_);
            assert( v.size() == size1_ );
            
            // Append the column
            append_column(v);

            // Move the column through the matrix to the right possition
            for(std::size_t k=size2_-1; k>j; ++k)
            {
                swap_columns(k,k-1);
            }

        }

        void swap_columns(std::size_t j1, std::size_t j2)
        {
            assert( j1 < size2_ && j2 < size2_ );
            std::swap_ranges(rows_begin(j1), rows_end(j1), rows_begin(j2) );
        }
        
        void swap_rows(std::size_t i1, std::size_t i2)
        {
            assert( i1 < size1_ && i2 < size1_ );
            // TODO find a better implementation
            std::swap_ranges( columns_begin(j1), columns_end(j1), columns_begin(j2) );
        }

        bool operator == (general_matrix const& rhs)
        {
            if(size1_ != rhs.size1_ || size2_ != rhs.size2_) return false;
            // TODO: reimplement - this is just a quick ugly implementation
            for(std::size_t j=0; j < size2_; ++j)
                for(std::size_t i=0; i< size1_; ++i)
                    if(operator()(i,j) != rhs(i,j)) return false;
            return true;
        }

        general_matrix<T>& operator += (general_matrix const& rhs) 
        {
            plus_assign(*this,rhs);
            return *this;
        }
        
        general_matrix<T>& operator -= (general_matrix const& rhs) 
        {
            minus_assign(*this,rhs);
            return *this;
        }
        
        general_matrix<T>& operator *= (T const& t)
        {
            multiplies_assign(*this, t);
            return *this;
        }

        // Default implementations
        void plus_assign(general_matrix const& rhs)
        {
            assert((rhs.size1() == size1_) && (rhs.size2() == size2_));
            if(!(this->is_shrinkable() || rhs.is_shrinkable()) )
            {
                std::transform(this->values_.begin(),this->values_.end(),rhs.values_.begin(),this->values_.begin(), std::plus<T>());
            }
            else
            {
                // Do the operation column by column
                for(std::size_t j=0; j < size2_; ++j)
                    std::transform(this->rows_begin(j), this->rows_end(j), rhs.rows_begin(j), this->rows_begin(j), std::plus<T>());
            }
        }

        void minus_assign(general_matrix const& rhs)
        {
            assert((rhs.size1() == size1_) && (rhs.size2() == size2_));
            if(!(this->is_shrinkable() || rhs.is_shrinkable()) )
            {
                std::transform(this->values_.begin(),this->values_.end(),rhs.values_.begin(),this->values_.begin(), std::minus<T>());
            }
            else
            {
                // Do the operation column by column
                for(std::size_t j=0; j < size2_; ++j)
                    std::transform(this->rows_begin(j), this->rows_end(j), rhs.rows_begin(j), this->rows_begin(j), std::minus<T>());
            }
        }
        
        void multiplies_assign (T const& t)
        {
            if(!(is_shrinkable()) )
            {
                std::transform(values_.begin(),values_.end(),values_.begin(), bind1st(std::multiplies<T>(),t));
            }
            else
            {
                // Do the operation column by column
                for(std::size_t j=0; j < size2_; ++j)
                    std::transform(rows_begin(j), rows_end(j), rows_begin(j), bind1st(std::multiplies<T>(),t));
            }
        }
        
        general_matrix<T>& operator *= (general_matrix const& rhs) 
        {
            // It's not common to implement a *= operator in terms of a * operator,
            // but a temporary object has to be created to store the result anyway
            // so it seems reasonable.
            general_matrix<T> tmp = (*this) * rhs;
            std::swap(tmp,*this);
            return *this;
        }

    private:
        friend class boost::numeric::bindings::detail::adaptor<general_matrix<T>,const general_matrix<T>, void>;
        friend class boost::numeric::bindings::detail::adaptor<general_matrix<T>,general_matrix<T>, void>;


        std::size_t size1_;
        std::size_t size2_;
        std::size_t reserved_size1_;
        // "reserved_size2_" is done automatically by underlying std::vector (see vector.reserve(), vector.capacity() )
        
        std::vector<T> values_;
    };
} // namespace blas

//
// An adaptor for the matrix to the boost::numeric::bindings
//
namespace boost { namespace numeric { namespace bindings { namespace detail {
        
    template <typename T, typename Id, typename Enable>
    struct adaptor< ::blas::general_matrix<T>, Id, Enable>
    {
        typedef typename copy_const< Id, T >::type              value_type;
        // TODO: fix the types of size and stride -> currently it's a workaround, since std::size_t causes problems with boost::numeric::bindings
        //typedef typename ::blas::general_matrix<T>::size_type         size_type;
        //typedef typename ::blas::general_matrix<T>::difference_type   difference_type;
        typedef std::ptrdiff_t  size_type;
        typedef std::ptrdiff_t  difference_type;

        typedef mpl::map<
            mpl::pair< tag::value_type,      value_type >,
            mpl::pair< tag::entity,          tag::matrix >,
            mpl::pair< tag::size_type<1>,    size_type >,
            mpl::pair< tag::size_type<2>,    size_type >,
            mpl::pair< tag::data_structure,  tag::linear_array >,
            mpl::pair< tag::data_order,      tag::column_major >,
            mpl::pair< tag::stride_type<1>,  tag::contiguous >,
            mpl::pair< tag::stride_type<2>,  difference_type >
        > property_map;

        static size_type size1( const Id& id ) {
            return id.num_rows();
        }

        static size_type size2( const Id& id ) {
            return id.num_columns();
        }

        static value_type* begin_value( Id& id ) {
            return &(*id.rows_begin(0));
        }

        static value_type* end_value( Id& id ) {
            return &(*(id.rows_end(id.num_columns()-1)-1));
        }

        static difference_type stride1( const Id& id ) {
            return 1;
        }

        static difference_type stride2( const Id& id ) {
           return id.reserved_size1_;
        }

    };
}}}}

//
// Hooked general matrix functions
//
namespace blas { namespace detail {

#define MATRIX_MATRIX_MULTIPLY(T) \
    const general_matrix<T> matrix_matrix_multiply(general_matrix<T> const& lhs, general_matrix<T> const& rhs) \
    { \
        assert( lhs.num_columns() == rhs.num_rows() ); \
        general_matrix<T> result(lhs.num_rows(),rhs.num_columns()); \
        boost::numeric::bindings::blas::gemm \
            ( \
               general_matrix<T>::value_type(1), \
               lhs, \
               rhs, \
               general_matrix<T>::value_type(1), \
               result \
            ); \
        return result; \
    }
IMPLEMENT_FOR_ALL_BLAS_TYPES(MATRIX_MATRIX_MULTIPLY)
#undef MATRIX_MATRIX_MULTIPLY

    template <typename T>
    const general_matrix<T> matrix_matrix_multiply(general_matrix<T> const& lhs, general_matrix<T> const& rhs)
    {
        assert( lhs.num_columns() == rhs.num_rows() );

        // Simple matrix matrix multiplication
        general_matrix<T> result(lhs.num_rows(),rhs.num_columns());
        for(std::size_t i=0; i < lhs.num_rows(); ++i)
        {
            for(std::size_t j=0; j<rhs.num_columns(); ++j)
            {
                for(std::size_t k=0; k<lhs.num_columns(); ++k)
                {
                        result(i,j) += lhs(i,k) * rhs(k,j);
                }
            }
        }
        return result;
    } 

// This seems to be the best solution for the *_ASSIGN dispatchers at the moment even though they call functions within the detail namespace
#define PLUS_MINUS_ASSIGN(T) \
    void plus_and_minus_assign_impl(general_matrix<T>, general_matrix<T> const& rhs, general_matrix<T>::value_type const& sign) \
    { \
        assert( m.num_columns() == rhs.num_columns() && m.num_rows() == rhs.num_rows() ); \
        if(!(m.is_shrinkable() || rhs.is_shrinkable()) ) \
        { \
            boost::numeric::bindings::blas::detail::axpy( m.num_rows() * m.num_columns(), sign, &(*rhs.rows_begin(0)), 1, &(*m.rows_begin(0)), 1); \
        } \
        else \
        { \
            for(std::size_t j=0; j < size2_; ++j) \
                boost::numeric::bindings::blas::detail::axpy( m.num_rows(), sign, &(*rhs.rows_begin(j)), 1, &(*m.rows_begin(j)), 1); \
        } \
    } \
    void plus_assign(general_matrix<T> m, general_matrix<T> const& rhs) \
    { \ plus_and_minus_assign_impl(m,rhs,general_matrix<T>::value_type(1)); } \
    void minus_assign(general_matrix<T> m, general_matrix<T> const& rhs) \
    { \ plus_and_minus_assign_impl(m,rhs,general_matrix<T>::value_type(-1)); }
IMPLEMENT_FOR_ALL_BLAS_TYPES(PLUS_MINUS_ASSIGN)
#undef PLUS_MINUS_ASSIGN

    template <typename T>
    void plus_assign(general_matrix<T>& m, general_matrix<T> const& rhs)
    {
        m.plus_assign(rhs);
    }

    template <typename T>
    void minus_assign(general_matrix<T>& m, general_matrix<T> const& rhs)
    {
        m.minus_assign(rhs);
    }


#define MULTIPLIES_ASSIGN(T) \
void multiplies_assign(general_matrix<T>& m, T const& t) \
    { \
        if( !(m.is_shrinkable()) ) \
        { \
            boost::numeric::bindings::blas::detail::scal( m.num_rows()*m.num_columns(), t, &(*m.rows_begin()), 1 ); \
        } \
        else \
        { \
            for(std::size_t j=0; j <m.num_columns(); ++j) \
                boost::numeric::bindings::blas::detail::scal( m.num_rows(), t, &(*m.rows_begin(j)), 1 ); \
        } \
    }
    IMPLEMENT_FOR_ALL_BLAS_TYPES(MULTIPLIES_ASSIGN)
#undef MULTIPLIES_ASSIGN


    template <typename T>    
    void multiplies_assign(general_matrix<T>& m, T const& t)
    {
        m.multiplies_assign(t);
    }

}} // namespace detail, namespace blas

//
// Free general matrix functions
//
namespace blas {
    template<typename T, class DoubleVector>
    void svd(general_matrix<T> M, general_matrix<T> & U, general_matrix<T> & V, DoubleVector & S)
    {
        std::size_t K = std::min(M.num_rows(), M.num_columns());
        U.resize(M.num_rows(), K);
        V.resize(K, M.num_columns());
        S.resize(K);
        
        boost::numeric::bindings::lapack::gesdd('S', M, S, U, V);
    }


    // 
    template <typename matrix_type>
    matrix_type transpose(matrix_type const& m) 
    {
        // TODO: perhaps this could return a proxy object
        matrix_type tmp(m.size2(),m.size1());
        for(std::size_t i=0;i<m.size1();++i){
            for(std::size_t j=0;j<m.size2();++j){
                tmp(j,i) = m()(i,j);
            }
        }
        return tmp;
    }
    
    template <typename matrix_type>
    const typename matrix_type::value_type trace(matrix_type const& m)
    {
        assert(m.size1() == m.size2());
        typename matrix_type::value_type tr = typename matrix_type::value_type(0);
        for(std::size_t i=0; i<m.size1(); ++i) tr+=m(i,i);
        return tr;
    }
        
    template<typename T>
    const vector<T> operator * (general_matrix<T> const& m, vector<T> const& v)
    {
        assert( m.size2() == v.size() );
        vector<T> result(m.size1());
        // Simple Matrix * Vector
        for(std::size_t i = 0; i < m.size1(); ++i)
        {
            for(std::size_t j=0; j <m.size2(); ++j)
            {
                result(i) = m(i,j) * v(j);
            }
        }
        return result;
    }
   
    // TODO: adj(Vector) * Matrix, where adj is a proxy object


    template<typename T>
    const general_matrix<T> operator * (general_matrix<T> m, T const& t)
    {
        return m*=t;
    }
    
    template<typename T>
    const general_matrix<T> operator * (T const& t,general_matrix<T> m)
    {
        return m*=t;
    }

    template<typename T>
    const general_matrix<T> operator * (general_matrix<T> const& m1, general_matrix<T> const& m2)
    {
        return matrix_matrix_multiply(m1,m2);
    }

    template<typename T>
    void gemm(general_matrix<T> const & A, general_matrix<T> const & B, general_matrix<T> & C)
    {
        C = matrix_matrix_multiply(A, B);
    }

    template <typename T>
    std::ostream& operator << (std::ostream& o, general_matrix<T> const& rhs)
    {
        for(std::size_t i=0; i< rhs.size1(); ++i)
        {
            for(std::size_t j=0; j < rhs.size2(); ++j)
                o<<rhs(i,j)<<" ";
            o<<std::endl;
        }
        return o;
    }
    
    
    
    
    
    
    
    
    
    
    
    
    
    /* make external
        T max() const
        {
            if(!is_shrinkable())
            {
                    return *max_element(values_.begin(), values_.end());
            }
            else
            {
                // Create a (shrinked) copy of the matrix and do the operation on the copy
                general_matrix tmp(*this);
                assert(tmp.is_shrinkable() == false);
                return tmp.max();
            }
        }
        
        T min() const
        {
            if(!is_shrinkable())
            {
                    return *min_element(values_.begin(), values_.end());
            }
            else
            {
                // Create a (shrinked) copy of the matrix and do the operation on the copy
                general_matrix tmp(*this);
                assert(tmp.is_shrinkable() == false);
                return tmp.min();
            }
        }
*/
} // namespace blas

#endif //__ALPS_GENERAL_MATRIX_HPP__
