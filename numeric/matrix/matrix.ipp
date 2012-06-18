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
        namespace detail {
            template <typename T, typename MemoryBlock, typename Operation>
            void op_assign_default_impl(dense_matrix<T,MemoryBlock>& lhs, dense_matrix<T,MemoryBlock> const& rhs, Operation op)
            {
                assert(lhs.num_rows() == rhs.num_rows());
                assert(lhs.num_cols() == rhs.num_cols());
                if(!(lhs.is_shrinkable() || rhs.is_shrinkable()) )
                {
                    std::transform(lhs.column(0).first,lhs.column(lhs.num_cols()-1).second,rhs.column(0).first,lhs.column(0).first, op);
                }
                else
                {
                    // Do the operation column by column
                    for(typename dense_matrix<T,MemoryBlock>::size_type j=0; j < lhs.num_cols(); ++j)
                    {
                        typedef typename dense_matrix<T,MemoryBlock>::column_element_iterator column_element_iterator;
                        std::pair<column_element_iterator,column_element_iterator> range(lhs.column(j));
                        std::transform( range.first, range.second, rhs.column(j).first, range.first, op);
                    }
                }
            }

        template <typename T, typename MemoryBlock, typename T2>
        void multiplies_assign_default_impl(dense_matrix<T,MemoryBlock>& lhs, T2 const& t)
        {
            if(!(lhs.is_shrinkable()) )
            {
                std::for_each(lhs.column(0).first, lhs.column(lhs.num_cols()-1).second, boost::lambda::_1 *= t);
            }
            else
            {
                // Do the operation column by column
                for(typename dense_matrix<T,MemoryBlock>::size_type j=0; j < lhs.num_cols(); ++j)
                {
                    typedef typename dense_matrix<T,MemoryBlock>::column_element_iterator column_element_iterator;
                    std::pair<column_element_iterator,column_element_iterator> range(lhs.column(j));
                    std::for_each(range.first, range.second, boost::lambda::_1 *= t);
                }
            }
        }
    } // end namespace detail

    template <typename T, typename MemoryBlock>
    dense_matrix<T, MemoryBlock> dense_matrix<T, MemoryBlock>::identity_matrix(size_type size)
    {
        dense_matrix<T, MemoryBlock> ret(size, size);
        for (size_type k = 0; k < size; ++k)
        ret(k,k) = 1;
        return ret;
    }

    template <typename T, typename MemoryBlock>
    dense_matrix<T, MemoryBlock>::dense_matrix(size_type rows, size_type cols, T init_value)
    : size1_(rows), size2_(cols), reserved_size1_(rows), values_(rows*cols, init_value)
    {
    }

    template <typename T, typename MemoryBlock>
    template <typename OtherMemoryBlock>
    dense_matrix<T, MemoryBlock>::dense_matrix(dense_matrix<T,OtherMemoryBlock> const& m)
    : size1_(m.size1_), size2_(m.size2_), reserved_size1_(m.size1_), values_()
    {
        // If the size of the matrix corresponds to the allocated size of the matrix...
        if(!m.is_shrinkable())
        {
            this->values_.insert(this->values_.end(), m.values_.begin(), m.values_.end());
        }
        else
        {
            // copy only a shrinked to size version of the original matrix
            this->values_.reserve(m.size1_*m.size2_);
            for(size_type j=0; j < m.size2_; ++j)
            {
                std::pair<typename dense_matrix<T,OtherMemoryBlock>::const_column_element_iterator,
                          typename dense_matrix<T,OtherMemoryBlock>::const_column_element_iterator
                         > range(m.column(j));
                this->values_.insert(this->values_.end(), range.first, range.second);
            }
        }
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::swap(dense_matrix & r)
    {
        using std::swap;
        swap(this->values_, r.values_);
        swap(this->size1_, r.size1_);
        swap(this->size2_, r.size2_);
        swap(this->reserved_size1_,r.reserved_size1_);
    }

    template <typename T, typename MemoryBlock>
    dense_matrix<T, MemoryBlock>& dense_matrix<T, MemoryBlock>::operator = (dense_matrix<T, MemoryBlock> rhs)
    {
        this->swap(rhs);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    inline T& dense_matrix<T, MemoryBlock>::operator()(const size_type i, const size_type j)
    {
        assert(i < this->size1_);
        assert(j < this->size2_);
        return this->values_[i+j*this->reserved_size1_];
    }

    template <typename T, typename MemoryBlock>
    inline T const& dense_matrix<T, MemoryBlock>::operator()(const size_type i, const size_type j) const 
    {
        assert((i < this->size1_) && (j < this->size2_));
        return this->values_[i+j*this->reserved_size1_];
    }

    template <typename T, typename MemoryBlock>
    bool dense_matrix<T, MemoryBlock>::operator == (dense_matrix const& rhs) const
    {
        if(this->size1_ != rhs.size1_ || this->size2_ != rhs.size2_) return false;
        // TODO: reimplement or remove - this is just a quick ugly implementation
        for(size_type i=0; i< this->size1_; ++i)
            for(size_type j=0; j < this->size2_; ++j)
                if(operator()(i,j) != rhs(i,j)) return false;
        return true;
    }

    template <typename T, typename MemoryBlock>
    dense_matrix<T,MemoryBlock>& dense_matrix<T, MemoryBlock>::operator += (dense_matrix const& rhs) 
    {
        plus_assign(*this,rhs);
        return *this;
    }
    
    template <typename T, typename MemoryBlock>
    dense_matrix<T,MemoryBlock>& dense_matrix<T, MemoryBlock>::operator -= (dense_matrix const& rhs) 
    {
        minus_assign(*this,rhs);
        return *this;
    }
    
    template <typename T, typename MemoryBlock>
    template <typename T2>
    dense_matrix<T,MemoryBlock>& dense_matrix<T, MemoryBlock>::operator *= (T2 const& t)
    {
        multiplies_assign(*this, t);
        return *this;
    }
    
    template <typename T, typename MemoryBlock>
    template <typename T2>
    dense_matrix<T,MemoryBlock>& dense_matrix<T, MemoryBlock>::operator /= (T2 const& t)
    {
        multiplies_assign(*this, T(1)/t);
        return *this;
    }

    template <typename T, typename MemoryBlock>
    inline bool dense_matrix<T, MemoryBlock>::empty() const
    {
        return (this->size1_ == 0 || this->size2_ == 0);
    }

    template <typename T, typename MemoryBlock>
    inline std::size_t dense_matrix<T, MemoryBlock>::num_rows() const
    {
        return this->size1_;
    }

    template <typename T, typename MemoryBlock>
    inline std::size_t dense_matrix<T, MemoryBlock>::num_cols() const
    {
        return this->size2_;
    }

    template <typename T, typename MemoryBlock>
    inline std::ptrdiff_t dense_matrix<T, MemoryBlock>::stride1() const
    {
        return 1;
    }

    template <typename T, typename MemoryBlock>
    inline std::ptrdiff_t dense_matrix<T, MemoryBlock>::stride2() const
    {
        return this->reserved_size1_;
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::resize(size_type size1, size_type size2, T const& init_value)
    {
        assert(size1 > 0);
        assert(size2 > 0);
        // Do we need more space? Reserve more space if needed!
        //
        // If the memory is reallocated using reserve
        // we just have to fill the new columns with the init_value
        // (->after this if statement),
        // since reserve fills all elements in the range between size1_
        // and reserved_size1_ of each EXISTING column with init_value
        // by using values_.resize()
        if(!automatic_reserve(size1,size2,init_value))
        {
            if(size1 > this->size1_)
            {
                // Reset all "new" elements which are in already reserved
                // rows of already existing columns to init_value
                // For all elements of new columns this is already done by
                // values_.resize() (->after this if statement)
                size_type num_of_cols = std::min(size2, this->size2_);
                for(size_type j=0; j < num_of_cols; ++j)
                    std::fill(
                            this->values_.begin()+j*this->reserved_size1_ + this->size1_,
                            this->values_.begin()+j*this->reserved_size1_ + size1,
                            init_value
                            );
            }
        }
        this->values_.resize(this->reserved_size1_*size2, init_value);
        this->size1_=size1;
        this->size2_=size2;
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::reserve(size_type size1, size_type size2, T const& init_value)
    {
        // The init_value may seem a little weird in a reserve method,
        // but one has to initialize all matrix elements in the
        // reserved_size1_ range of each column, due to the 1d-structure
        // of the underlying MemoryBlock (e.g. std::vector)

        // Ignore values that would shrink the matrix
        size2 = std::max(size2, this->size2_);
        size1 = std::max(size1, this->reserved_size1_);
       
        // Is change of structure or size of the MemoryBlock necessary?
        if(size1 > this->reserved_size1_ || size1*size2 > this->values_.capacity() )
        {
            MemoryBlock tmp;
            tmp.reserve(size1*size2);
            // Copy column by column
            for(size_type j=0; j < this->size2_; ++j)
            {
                std::pair<column_element_iterator, column_element_iterator> range(column(j));
                // Copy the elements from the current MemoryBlock
                tmp.insert(tmp.end(),range.first,range.second);
                // and fill the rest with the init_value
                tmp.insert(tmp.end(),size1-this->size1_,init_value);
            }
            std::swap(this->values_,tmp);
            this->reserved_size1_ = size1;
        }
    }

    template <typename T, typename MemoryBlock>
    std::pair<std::size_t,std::size_t> dense_matrix<T, MemoryBlock>::capacity() const
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
    bool dense_matrix<T, MemoryBlock>::is_shrinkable() const
    {
        // This assertion should actually never fail
        assert( this->reserved_size1_*this->size2_ == this->values_.size() );
        if(this->size1_ < this->reserved_size1_) return true;
        else return false;
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::clear()
    {
        // Clear the values vector and ensure the reserved size stays the way it was
        this->values_.clear();
        this->values_.resize(this->reserved_size1_*this->size2_);
        this->size1_ = 0;
        this->size2_ = 0;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void dense_matrix<T, MemoryBlock>::append_cols(std::pair<InputIterator,InputIterator> const& range, difference_type k)
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
    void dense_matrix<T, MemoryBlock>::append_rows(std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        assert( std::distance(range.first, range.second) == k*this->size2_ );
        // Reserve more space if needed
        automatic_reserve(this->size1_+k, this->size2_);
        // The elements do already exists due to reserve, so we can just use (copy to) them.
        for(difference_type l=0; l<k; ++l)
            std::copy( range.first+(l*this->size2_), range.first+((l+1)*this->size2_), row(this->size1_+l).first );
        this->size1_ += k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void dense_matrix<T, MemoryBlock>::insert_rows(size_type i, std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        assert( i <= this->size1_ );
        assert( std::distance(range.first, range.second) == k*this->size2_ );

        // Append the row
        automatic_reserve(this->size1_+k,this->size2_);

        for(size_type j=0; j<this->size2_; ++j)
            std::copy_backward(&this->values_[this->reserved_size1_*j+i],&this->values_[this->reserved_size1_*j+this->size1_],&this->values_[this->reserved_size1_*j+this->size1_+k]);
        for(difference_type l=0; l<k; ++l)
            std::copy(range.first+l*this->size2_,range.first+(l+1)*this->size2_,row(i+l).first);
        this->size1_+=k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void dense_matrix<T, MemoryBlock>::insert_cols(size_type j, std::pair<InputIterator,InputIterator> const& range, difference_type k)
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
    void dense_matrix<T, MemoryBlock>::remove_rows(size_type i, difference_type k)
    {
        assert( i+k <= this->size1_ );
        // for each column, copy the rows > i+k   k rows  up
        for(size_type j = 0; j < this->size2_; ++j)
            std::copy(&this->values_[this->reserved_size1_*j + i + k], &this->values_[this->reserved_size1_*j + this->size1_], &this->values_[this->reserved_size1_*j + i] );
        this->size1_ -= k;
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::remove_cols(size_type j, difference_type k)
    {
        assert( j+k <= this->size2_ );
        this->values_.erase(this->values_.begin()+(this->reserved_size1_*j), this->values_.begin()+(this->reserved_size1_*(j+k)) );
        this->size2_ -= k;
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::swap_rows(size_type i1, size_type i2)
    {
        assert( i1 < this->size1_ && i2 < this->size1_ );
        std::pair<row_element_iterator, row_element_iterator> range( row(i1) );
        std::swap_ranges( range.first, range.second, row(i2).first );
    }

    template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::swap_cols(size_type j1, size_type j2)
    {
        assert( j1 < this->size2_ && j2 < this->size2_ );
        std::pair<column_element_iterator, column_element_iterator> range( column(j1) );
        std::swap_ranges(range.first, range.second, column(j2).first );
    }

// TODO
//    template <typename T, typename MemoryBlock>
//    void dense_matrix<T, MemoryBlock>::inplace_conjugate()
//    {
//		// Do the operation column by column
//		for(size_type j=0; j < this->size2_; ++j)
//		{
//			std::pair<column_element_iterator,column_element_iterator> range(column(j));
//			std::transform(range.first, range.second,
//						   range.first, utils::functor_conj());
//		}
//    }

    template <typename T, typename MemoryBlock>
    inline bool dense_matrix<T, MemoryBlock>::automatic_reserve(size_type size1, size_type size2, T const& init_value)
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

#ifdef HAVE_ALPS_HDF5
	template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::load(alps::hdf5::archive & ar) {
        load_impl(ar, typename alps::is_complex<T>::type());
    }
	template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::load_impl(alps::hdf5::archive & ar, boost::mpl::true_)
    {
        ar >> alps::make_pvp("size1", size1_);
        ar >> alps::make_pvp("size2", size2_);
        ar >> alps::make_pvp("reserved_size1", reserved_size1_);
        if (ar.is_complex("values"))
            ar >> alps::make_pvp("values", values_);    
        else {
            std::vector<typename T::value_type> data;
            ar >> alps::make_pvp("values", data);
            values_.resize(data.size());
            std::copy(data.begin(), data.end(), values_.begin());
        }
    }
	template <typename T, typename MemoryBlock>
    void dense_matrix<T, MemoryBlock>::load_impl(alps::hdf5::archive & ar, boost::mpl::false_)
    {
        ar >> alps::make_pvp("size1", size1_);
        ar >> alps::make_pvp("size2", size2_);
        ar >> alps::make_pvp("reserved_size1", reserved_size1_);
        ar >> alps::make_pvp("values", values_);
    }

	template <typename T, typename MemoryBlock>
    inline void dense_matrix<T, MemoryBlock>::save(alps::hdf5::archive & ar) const
    {
		ar << alps::make_pvp("size1", size1_);
		ar << alps::make_pvp("size2", size2_);
		ar << alps::make_pvp("reserved_size1", reserved_size1_);
		ar << alps::make_pvp("values", values_);
    }
#endif	

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> matrix_matrix_multiply(dense_matrix<T,MemoryBlock> const& lhs, dense_matrix<T,MemoryBlock> const& rhs)
    {
        assert( !(lhs.num_cols() > rhs.num_rows()) );
        assert( !(lhs.num_cols() < rhs.num_rows()) );
        assert( lhs.num_cols() == rhs.num_rows() );
        
        // Simple matrix matrix multiplication
        dense_matrix<T,MemoryBlock> result(lhs.num_rows(),rhs.num_cols());
        for(std::size_t i=0; i < lhs.num_rows(); ++i)
        {
            for(std::size_t k=0; k<lhs.num_cols(); ++k)
            {
                for(std::size_t j=0; j<rhs.num_cols(); ++j)
                {
                        result(i,j) += lhs(i,k) * rhs(k,j);
                }
            }
        }
        
        return result;
    } 

    template<typename T, typename MemoryBlock, typename T2, typename MemoryBlock2>
    const vector<typename MultiplyReturnType<T,MemoryBlock,T2,MemoryBlock2>::value_type,typename MultiplyReturnType<T,MemoryBlock,T2,MemoryBlock2>::memoryblock_type>
    matrix_vector_multiply(dense_matrix<T,MemoryBlock> const& m, vector<T2,MemoryBlock2> const& v)
    {
        assert( m.num_cols() == v.size() );
        vector<
            typename MultiplyReturnType<T,MemoryBlock,T2,MemoryBlock2>::value_type,
            typename MultiplyReturnType<T,MemoryBlock,T2,MemoryBlock2>::memoryblock_type
            >
            result(m.num_rows());
        // Simple Matrix * Vector
        for(typename dense_matrix<T,MemoryBlock>::size_type i = 0; i < m.num_rows(); ++i)
        {
            for(typename dense_matrix<T,MemoryBlock>::size_type j=0; j <m.num_cols(); ++j)
            {
                result(i) += m(i,j) * v(j);
            }
        }
        return result;
    }

    template <typename T,typename MemoryBlock>
    void plus_assign(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs)
    {
        detail::op_assign_default_impl(m,rhs,std::plus<T>());
    }

    template <typename T, typename MemoryBlock>
    void minus_assign(dense_matrix<T,MemoryBlock>& m, dense_matrix<T,MemoryBlock> const& rhs)
    {
        detail::op_assign_default_impl(m,rhs,std::minus<T>());
    }

    template <typename T, typename MemoryBlock, typename T2>
    void multiplies_assign(dense_matrix<T,MemoryBlock>& m, T2 const& t)
    {
        detail::multiplies_assign_default_impl(m,t);
    }

//////////////////////////////////////////////////////////////////////////////

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator + (dense_matrix<T,MemoryBlock> a, dense_matrix<T,MemoryBlock> const& b)
    {
        a += b;
        return a;
    }

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator - (dense_matrix<T,MemoryBlock> a, dense_matrix<T,MemoryBlock> const& b)
    {
        a -= b;
        return a;
    }

    template <typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator - (dense_matrix<T,MemoryBlock> a)
    {
		// Do the operation column by column
		for(typename dense_matrix<T,MemoryBlock>::size_type j=0; j < a.num_cols(); ++j)
		{
			std::pair<typename dense_matrix<T,MemoryBlock>::column_element_iterator,
					  typename dense_matrix<T,MemoryBlock>::column_element_iterator> range(a.column(j));
			std::transform(range.first, range.second,
						   range.first, std::negate<T>());
		}
		return a;
    }
	
    template<typename T, typename MemoryBlock, typename T2, typename MemoryBlock2>
    const vector<typename MultiplyReturnType<T,MemoryBlock,T2,MemoryBlock2>::value_type, typename MultiplyReturnType<T,MemoryBlock,T2,MemoryBlock2>::memoryblock_type>
    operator * (dense_matrix<T,MemoryBlock> const& m, vector<T2,MemoryBlock2> const& v)
    {
        return matrix_vector_multiply(m,v);
    }

    template<typename T,typename MemoryBlock, typename T2>
    const dense_matrix<T,MemoryBlock> operator * (dense_matrix<T,MemoryBlock> m, T2 const& t)
    {
        return m*=t;
    }

    template<typename T,typename MemoryBlock, typename T2>
    const dense_matrix<T,MemoryBlock> operator * (T2 const& t, dense_matrix<T,MemoryBlock> m)
    {
        return m*=t;
    }

    template<typename T, typename MemoryBlock>
    const dense_matrix<T,MemoryBlock> operator * (dense_matrix<T,MemoryBlock> const& m1, dense_matrix<T,MemoryBlock> const& m2)
    {
        return matrix_matrix_multiply(m1,m2);
    }

    template<typename T,typename MemoryBlock>
    void gemm(dense_matrix<T,MemoryBlock> const & A, dense_matrix<T,MemoryBlock> const & B, dense_matrix<T,MemoryBlock> & C)
    {
        C = matrix_matrix_multiply(A, B);
    }
    
    template<class T, class MemoryBlock>
    std::size_t size_of(dense_matrix<T, MemoryBlock> const & m)
    {
        return num_rows(m)*num_cols(m)*sizeof(T);
    }
    
    template <typename T, typename MemoryBlock>
    std::ostream& operator << (std::ostream& o, dense_matrix<T,MemoryBlock> const& m)
    {
        for(typename dense_matrix<T,MemoryBlock>::size_type i=0; i< m.num_rows(); ++i)
        {
            for(typename dense_matrix<T,MemoryBlock>::size_type j=0; j < m.num_cols(); ++j)
                o<<m(i,j)<<" ";
            o<<std::endl;
        }
        return o;
    }

   template <typename T, typename MemoryBlock>
   const MemoryBlock& dense_matrix<T, MemoryBlock>::get_values() const
   {   
       return this->values_;
   }   

   template <typename T, typename MemoryBlock>
   MemoryBlock& dense_matrix<T, MemoryBlock>::get_values()
   {   
       return this->values_;
   }   

   } // end namespace numeric
} // end namespace alps
