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

#include <alps/numeric/conj.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/iterator/transform_iterator.hpp>
#include <alps/utility/numeric_cast.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace alps {
    namespace numeric {
        namespace detail {

            template <typename T, typename T2>
            struct insert_cast_helper
            {
                template <typename MemoryBlock, typename InputIterator>
                static void apply(MemoryBlock& mem, InputIterator it, InputIterator end)
                {
                    mem.insert(
                        mem.end()
                        , boost::make_transform_iterator( it, numeric_cast<T,T2>)
                        , boost::make_transform_iterator(end, numeric_cast<T,T2>)
                    );
                }
            };

            template <typename T>
            struct insert_cast_helper<T,T>
            {
                template <typename MemoryBlock, typename InputIterator>
                static void apply(MemoryBlock& mem, InputIterator it, InputIterator end)
                {
                    mem.insert( mem.end() , it, end);
                }
            };

            template <typename T, typename MemoryBlock, typename Operation>
            void op_assign_default_impl(matrix<T,MemoryBlock>& lhs, matrix<T,MemoryBlock> const& rhs, Operation op)
            {
                assert(lhs.num_rows() == rhs.num_rows());
                assert(lhs.num_cols() == rhs.num_cols());
                if(!(lhs.is_shrinkable() || rhs.is_shrinkable()) )
                {
                    std::transform(lhs.col(0).first,lhs.col(lhs.num_cols()-1).second,rhs.col(0).first,lhs.col(0).first, op);
                }
                else
                {
                    // Do the operation column by column
                    for(typename matrix<T,MemoryBlock>::size_type j=0; j < lhs.num_cols(); ++j)
                    {
                        typedef typename matrix<T,MemoryBlock>::col_element_iterator col_element_iterator;
                        std::pair<col_element_iterator,col_element_iterator> range(lhs.col(j));
                        std::transform( range.first, range.second, rhs.col(j).first, range.first, op);
                    }
                }
            }

        template <typename T, typename MemoryBlock, typename T2>
        void multiplies_assign_default_impl(matrix<T,MemoryBlock>& lhs, T2 const& t)
        {
            if(!(lhs.is_shrinkable()) )
            {
                std::for_each(lhs.col(0).first, lhs.col(lhs.num_cols()-1).second, boost::lambda::_1 *= t);
            }
            else
            {
                // Do the operation column by column
                for(typename matrix<T,MemoryBlock>::size_type j=0; j < lhs.num_cols(); ++j)
                {
                    typedef typename matrix<T,MemoryBlock>::col_element_iterator col_element_iterator;
                    std::pair<col_element_iterator,col_element_iterator> range(lhs.col(j));
                    std::for_each(range.first, range.second, boost::lambda::_1 *= t);
                }
            }
        }
    } // end namespace detail

    template <typename T, typename MemoryBlock>
    matrix<T, MemoryBlock> matrix<T, MemoryBlock>::identity_matrix(size_type size)
    {
        matrix<T, MemoryBlock> ret(size, size);
        for (size_type k = 0; k < size; ++k)
            ret(k,k) = 1;
        return ret;
    }

    template <typename T, typename MemoryBlock>
    matrix<T, MemoryBlock>::matrix(size_type rows, size_type cols, T init_value)
    : size1_(rows), size2_(cols), reserved_size1_(rows), values_(rows*cols, init_value)
    {
    }

    template <typename T, typename MemoryBlock>
    matrix<T, MemoryBlock>::matrix(matrix const& m)
    : size1_(m.size1_), size2_(m.size2_), reserved_size1_(m.size1_), values_(copy_values(m))
    { }

    template <typename T, typename MemoryBlock>
    template <typename T2, typename OtherMemoryBlock>
    matrix<T, MemoryBlock>::matrix(matrix<T2,OtherMemoryBlock> const& m)
    : size1_(m.size1_), size2_(m.size2_), reserved_size1_(m.size1_), values_(copy_values(m))
    { }

    template <typename T, typename MemoryBlock>
    template <typename ForwardIterator>
    matrix<T, MemoryBlock>::matrix(std::vector<std::pair<ForwardIterator,ForwardIterator> > const& columns)
    : size1_(0), size2_(0), reserved_size1_(0), values_()
    {
        using std::distance;
        using std::copy;
        assert(columns.size() > 0);

        size_type const reserve_rows = distance(columns.front().first, columns.front().second);
        values_.reserve(reserve_rows*columns.size());
        for(std::size_t i=0; i < columns.size(); ++i) {
            assert(distance(columns[i].first,columns[i].second) == reserve_rows);
            copy(columns[i].first, columns[i].second, std::back_inserter(values_));
        }
        reserved_size1_ = reserve_rows;
        size1_ = reserve_rows;
        size2_ = columns.size();
    }

    template <typename T, typename MemoryBlock>
    template <typename T2, typename OtherMemoryBlock>
    MemoryBlock matrix<T, MemoryBlock>::copy_values(matrix<T2,OtherMemoryBlock> const& m)
    {
        MemoryBlock ret;
        // If the size of the matrix corresponds to the allocated size of the matrix...
        if(!m.is_shrinkable())
        {
            detail::insert_cast_helper<T,T2>::apply(ret, m.values_.begin(), m.values_.end());
        }
        else
        {
            // copy only a shrinked to size version of the original matrix
            ret.reserve(m.size1_*m.size2_);
            for(size_type j=0; j < m.size2_; ++j)
            {
                std::pair<typename matrix<T2,OtherMemoryBlock>::const_col_element_iterator,
                          typename matrix<T2,OtherMemoryBlock>::const_col_element_iterator
                         > range(m.col(j));
                detail::insert_cast_helper<T,T2>::apply(ret, range.first, range.second);
            }
        }
        return ret;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::swap(matrix & r)
    {
        using std::swap;
        swap(this->values_, r.values_);
        swap(this->size1_, r.size1_);
        swap(this->size2_, r.size2_);
        swap(this->reserved_size1_,r.reserved_size1_);
    }

    template <typename T, typename MemoryBlock>
    matrix<T, MemoryBlock>& matrix<T, MemoryBlock>::operator = (matrix<T, MemoryBlock> rhs)
    {
        this->swap(rhs);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    inline T& matrix<T, MemoryBlock>::operator()(const size_type i, const size_type j)
    {
        assert(i < this->size1_);
        assert(j < this->size2_);
        return this->values_[i+j*this->reserved_size1_];
    }

    template <typename T, typename MemoryBlock>
    inline T const& matrix<T, MemoryBlock>::operator()(const size_type i, const size_type j) const
    {
        assert((i < this->size1_) && (j < this->size2_));
        return this->values_[i+j*this->reserved_size1_];
    }

    template <typename T, typename MemoryBlock>
    bool matrix<T, MemoryBlock>::operator == (matrix const& rhs) const
    {
        // TODO this is not really good for floats and doubles
        using std::equal;
        if(this->size1_ != rhs.size1_ || this->size2_ != rhs.size2_)
            return false;
        for(size_type j=0; j < this->size2_; ++j)
            if( !equal(this->col(j).first, this->col(j).second, rhs.col(j).first) )
                return false;
        return true;
    }

    template <typename T, typename MemoryBlock>
    matrix<T,MemoryBlock>& matrix<T, MemoryBlock>::operator += (matrix const& rhs)
    {
        plus_assign(*this,rhs);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    matrix<T,MemoryBlock>& matrix<T, MemoryBlock>::operator -= (matrix const& rhs)
    {
        minus_assign(*this,rhs);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    template <typename T2>
    matrix<T,MemoryBlock>& matrix<T, MemoryBlock>::operator *= (T2 const& t)
    {
        multiplies_assign(*this, t);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    template <typename T2>
    matrix<T,MemoryBlock>& matrix<T, MemoryBlock>::operator /= (T2 const& t)
    {
        multiplies_assign(*this, T(1)/t);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    inline bool matrix<T, MemoryBlock>::empty() const
    {
        return (this->size1_ == 0 || this->size2_ == 0);
    }

    template <typename T, typename MemoryBlock>
    inline std::size_t matrix<T, MemoryBlock>::num_rows() const
    {
        return this->size1_;
    }

    template <typename T, typename MemoryBlock>
    inline std::size_t matrix<T, MemoryBlock>::num_cols() const
    {
        return this->size2_;
    }

    template <typename T, typename MemoryBlock>
    inline std::ptrdiff_t matrix<T, MemoryBlock>::stride1() const
    {
        return 1;
    }

    template <typename T, typename MemoryBlock>
    inline std::ptrdiff_t matrix<T, MemoryBlock>::stride2() const
    {
        return this->reserved_size1_;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::resize(size_type rows, size_type cols, T const& init_value)
    {
        assert(rows > 0);
        assert(cols > 0);
        // Do we need more space? Reserve more space if needed!
        //
        // If the memory is reallocated using reserve
        // we just have to fill the new columns with the init_value
        // (->after this if statement),
        // since reserve fills all elements in the range between size1_
        // and reserved_size1_ of each EXISTING column with init_value
        // by using values_.resize()
        if(!automatic_reserve(rows,cols,init_value))
        {
            if(rows > this->size1_)
            {
                // Reset all "new" elements which are in already reserved
                // rows of already existing columns to init_value
                // For all elements of new columns this is already done by
                // values_.resize() (->after this if statement)
                size_type num_of_cols = (std::min)(cols, this->size2_);
                for(size_type j=0; j < num_of_cols; ++j)
                    std::fill(
                            this->values_.begin()+j*this->reserved_size1_ + this->size1_,
                            this->values_.begin()+j*this->reserved_size1_ + rows,
                            init_value
                            );
            }
        }
        this->values_.resize(this->reserved_size1_*cols, init_value);
        this->size1_=rows;
        this->size2_=cols;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::reserve(size_type rows, size_type cols, T const& init_value)
    {
        // The init_value may seem a little weird in a reserve method,
        // but one has to initialize all matrix elements in the
        // reserved_size1_ range of each column, due to the 1d-structure
        // of the underlying MemoryBlock (e.g. std::vector)
        using std::swap;
        // Ignore values that would shrink the matrix
        cols = (std::max)(cols, this->size2_);
        rows = (std::max)(rows, this->reserved_size1_);

        // Is change of structure or size of the MemoryBlock necessary?
        if(rows > this->reserved_size1_ || rows*cols > this->values_.capacity() )
        {
            MemoryBlock tmp;
            tmp.reserve(rows*cols);
            // Copy column by column
            for(size_type j=0; j < this->size2_; ++j)
            {
                std::pair<col_element_iterator, col_element_iterator> range(col(j));
                // Copy the elements from the current MemoryBlock
                tmp.insert(tmp.end(),range.first,range.second);
                // and fill the rest with the init_value
                tmp.insert(tmp.end(),rows-this->size1_,init_value);
            }
            swap(this->values_,tmp);
            this->reserved_size1_ = rows;
        }
    }

    template <typename T, typename MemoryBlock>
    std::pair<std::size_t,std::size_t> matrix<T, MemoryBlock>::capacity() const
    {
        assert( this->values_.capacity() % this->reserved_size1_ == 0 );
        // Evaluate the maximal number of columns (with size reserved_size1_) that the underlying vector could hold.
        // If the constructor, resize() and reserve() of std::vector would guarantee to allocate 
        // the requested amount of memory exactly
        // values_.capacity() % reserved_size1_ == 0 should hold.
        // However these functions guarantee to allocate _at least_ the requested amount.
        size_type reserved_size2_ = this->values_.capacity() - (this->values_.capacity() % this->reserved_size1_) / this->reserved_size1_;
        return std::pair<size_type,size_type>( this->reserved_size1_, reserved_size2_ );
    }

    template <typename T, typename MemoryBlock>
    bool matrix<T, MemoryBlock>::is_shrinkable() const
    {
        // This assertion should actually never fail
        assert( this->reserved_size1_*this->size2_ == this->values_.size() );
        if(this->size1_ < this->reserved_size1_) return true;
        else return false;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::clear()
    {
        // Clear the values vector and ensure the reserved size stays the way it was
        this->values_.clear();
        this->values_.resize(this->reserved_size1_*this->size2_);
        this->size1_ = 0;
        this->size2_ = 0;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::append_cols(std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        assert( std::distance(range.first, range.second) == k*this->size1_ );
        // Reserve more space if needed
        automatic_reserve(this->size1_,this->size2_+k);
        // Append column by column
        for(difference_type l=0; l<k; ++l)
        {
            this->values_.insert(this->values_.end(), range.first+(l*this->size1_), range.first+((l+1)*this->size1_) );
            // Fill the space reserved for new rows
            this->values_.insert(this->values_.end(), this->reserved_size1_-this->size1_, T());
        }
        this->size2_ += k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::append_rows(std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        assert( std::distance(range.first, range.second) == k*this->size2_ );
        // Reserve more space if needed
        automatic_reserve(this->size1_+k, this->size2_);
        // The elements do already exists due to reserve, so we can just use (copy to) them.
        for(difference_type l=0; l<k; ++l)
            std::copy( range.first+(l*this->size2_), range.first+((l+1)*this->size2_), row_element_iterator(&values_[size1_+l],reserved_size1_) );
        this->size1_ += k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::insert_rows(size_type i, std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        assert( i <= this->size1_ );
        assert( std::distance(range.first, range.second) == k*this->size2_ );

        // Append the row
        automatic_reserve(this->size1_+k,this->size2_);

        for(size_type j=0; j<this->size2_; ++j)
            std::copy_backward(&this->values_[this->reserved_size1_*j+i],&this->values_[this->reserved_size1_*j+this->size1_],&this->values_[this->reserved_size1_*j+this->size1_+k]);
        for(difference_type l=0; l<k; ++l)
            std::copy(range.first+l*this->size2_,range.first+(l+1)*this->size2_,row_element_iterator(&values_[i+l],reserved_size1_) );
        this->size1_+=k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::insert_cols(size_type j, std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        assert( j <= this->size2_);
        assert( std::distance(range.first, range.second) == k*this->size1_ );

        // Append the column
        automatic_reserve(this->size1_,this->size2_+k);

        // Move the column through the matrix to the right possition
        for(size_type h=this->size2_; h>j; --h)
            std::copy(&this->values_[this->reserved_size1_*(h-1)],&this->values_[this->reserved_size1_*(h-1)]+this->size1_,&this->values_[this->reserved_size1_*(h+k-1)]);
        for(difference_type l=0; l<k; ++l)
            std::copy(range.first+l*this->size1_,range.first+(l+1)*this->size1_,&this->values_[this->reserved_size1_*(j+l)]);
        this->size2_+=k;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::remove_rows(size_type i, difference_type k)
    {
        assert( i+k <= this->size1_ );
        // for each column, copy the rows > i+k   k rows  up
        for(size_type j = 0; j < this->size2_; ++j)
            std::copy(&this->values_[this->reserved_size1_*j + i + k], &this->values_[this->reserved_size1_*j + this->size1_], &this->values_[this->reserved_size1_*j + i] );
        this->size1_ -= k;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::remove_cols(size_type j, difference_type k)
    {
        assert( j+k <= this->size2_ );
        this->values_.erase(this->values_.begin()+(this->reserved_size1_*j), this->values_.begin()+(this->reserved_size1_*(j+k)) );
        this->size2_ -= k;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::swap_rows(size_type i1, size_type i2)
    {
        assert( i1 < this->size1_ && i2 < this->size1_ );
        std::pair<row_element_iterator, row_element_iterator> range( row(i1) );
        std::swap_ranges( range.first, range.second, row(i2).first );
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::swap_cols(size_type j1, size_type j2)
    {
        assert( j1 < this->size2_ && j2 < this->size2_ );
        std::pair<col_element_iterator, col_element_iterator> range( col(j1) );
        std::swap_ranges(range.first, range.second, col(j2).first );
    }

    template <typename T, typename MemoryBlock>
    inline bool matrix<T, MemoryBlock>::automatic_reserve(size_type size1, size_type size2, T const& init_value)
    {
        // Do we need to reserve more space in any dimension?
        if(size1 > this->reserved_size1_ || this->reserved_size1_*size2 > this->values_.capacity())
        {
            reserve(size1*3/2,size2*3/2,init_value);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::write_xml(oxstream& xml) const
    {
        xml << start_tag("MATRIX");
        xml << attribute("cols", num_cols());
        xml << attribute("rows", num_rows());
        for(size_type i=0; i < num_rows(); ++i)
        {
            xml << start_tag("ROW");
            for(size_type j=0; j < num_cols(); ++j)
            {
                std::stringstream sts;
                sts << this->operator()(i,j);

                xml << start_tag("ELEMENT");
                xml << sts.str();
                xml << end_tag("ELEMENT");
            }
            xml << end_tag("ROW");
        }
        xml << end_tag("MATRIX");
    }

    template<typename T, typename MemoryBlock, typename T2, typename MemoryBlock2>
    typename matrix_vector_multiplies_return_type<matrix<T,MemoryBlock>,vector<T2,MemoryBlock2> >::type
    matrix_vector_multiply(matrix<T,MemoryBlock> const& m, vector<T2,MemoryBlock2> const& v)
    {
        assert( m.num_cols() == v.size() );
        typename matrix_vector_multiplies_return_type<matrix<T,MemoryBlock>,vector<T2,MemoryBlock2> >::type
            result(m.num_rows());
        // Simple Matrix * Vector
        for(typename matrix<T,MemoryBlock>::size_type i = 0; i < m.num_rows(); ++i)
        {
            for(typename matrix<T,MemoryBlock>::size_type j=0; j <m.num_cols(); ++j)
            {
                result(i) += m(i,j) * v(j);
            }
        }
        return result;
    }

    template <typename T,typename MemoryBlock>
    void plus_assign(matrix<T,MemoryBlock>& m, matrix<T,MemoryBlock> const& rhs)
    {
        detail::op_assign_default_impl(m,rhs,std::plus<T>());
    }

    template <typename T, typename MemoryBlock>
    void minus_assign(matrix<T,MemoryBlock>& m, matrix<T,MemoryBlock> const& rhs)
    {
        detail::op_assign_default_impl(m,rhs,std::minus<T>());
    }

    template <typename T, typename MemoryBlock, typename T2>
    void multiplies_assign(matrix<T,MemoryBlock>& m, T2 const& t)
    {
        detail::multiplies_assign_default_impl(m,t);
    }

//////////////////////////////////////////////////////////////////////////////

    template <typename T, typename MemoryBlock>
    const matrix<T,MemoryBlock> operator + (matrix<T,MemoryBlock> a, matrix<T,MemoryBlock> const& b)
    {
        a += b;
        return a;
    }

    template <typename T, typename MemoryBlock>
    const matrix<T,MemoryBlock> operator - (matrix<T,MemoryBlock> a, matrix<T,MemoryBlock> const& b)
    {
        a -= b;
        return a;
    }

    template <typename T, typename MemoryBlock>
    const matrix<T,MemoryBlock> operator - (matrix<T,MemoryBlock> a)
    {
        // Do the operation column by column
        for(typename matrix<T,MemoryBlock>::size_type j=0; j < a.num_cols(); ++j)
        {
            std::pair<typename matrix<T,MemoryBlock>::col_element_iterator,
                typename matrix<T,MemoryBlock>::col_element_iterator> range(a.col(j));
            std::transform(range.first, range.second,range.first, std::negate<T>());
        }
        return a;
    }

    template<typename T, typename MemoryBlock, typename T2, typename MemoryBlock2>
    typename matrix_vector_multiplies_return_type<matrix<T,MemoryBlock>,vector<T2,MemoryBlock2> >::type
    operator * (matrix<T,MemoryBlock> const& m, vector<T2,MemoryBlock2> const& v)
    {
        return matrix_vector_multiply(m,v);
    }

    template<typename T,typename MemoryBlock, typename T2>
    typename boost::enable_if<is_matrix_scalar_multiplication<matrix<T,MemoryBlock>,T2>, matrix<T,MemoryBlock> >::type operator * (matrix<T,MemoryBlock> m, T2 const& t)
    {
        return m*=t;
    }

    template<typename T,typename MemoryBlock, typename T2>
    typename boost::enable_if<is_matrix_scalar_multiplication<matrix<T,MemoryBlock>,T2>, matrix<T,MemoryBlock> >::type operator * (T2 const& t, matrix<T,MemoryBlock> m)
    {
        return m*=t;
    }

    template<typename T, typename MemoryBlock>
    const matrix<T,MemoryBlock> operator * (matrix<T,MemoryBlock> const& m1, matrix<T,MemoryBlock> const& m2)
    {
        return matrix_matrix_multiply(m1,m2);
    }

    template<class T, class MemoryBlock>
    std::size_t size_of(matrix<T, MemoryBlock> const & m)
    {
        return num_rows(m)*num_cols(m)*sizeof(T);
    }

    template <typename T, typename MemoryBlock>
    std::ostream& operator << (std::ostream& o, matrix<T,MemoryBlock> const& m)
    {
        for(typename matrix<T,MemoryBlock>::size_type i=0; i< m.num_rows(); ++i)
        {
            for(typename matrix<T,MemoryBlock>::size_type j=0; j < m.num_cols(); ++j)
                o<<m(i,j)<<" ";
            o<<std::endl;
        }
        return o;
    }

    template <typename T, typename MemoryBlock>
    alps::oxstream& operator << (alps::oxstream& xml, matrix<T,MemoryBlock> const& m)
    {
        m.write_xml(xml);
        return xml;
    }

   template <typename T, typename MemoryBlock>
   const MemoryBlock& matrix<T, MemoryBlock>::get_values() const
   {
       return this->values_;
   }

   template <typename T, typename MemoryBlock>
   MemoryBlock& matrix<T, MemoryBlock>::get_values()
   {
       return this->values_;
   }

   template <typename T, typename MemoryBlock>
   template<typename Archive> 
   inline void matrix<T, MemoryBlock>::serialize(Archive & ar, const unsigned int version)
   {
       ar & size1_ & size2_ & reserved_size1_ & values_;
   }

   } // end namespace numeric
} // end namespace alps
