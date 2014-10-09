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
#include <boost/iterator/transform_iterator.hpp>
#include <alps/utilities/numeric_cast.hpp>

#include <boost/serialization/serialization.hpp>
#include <boost/serialization/vector.hpp>

namespace alps {
namespace numeric {

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
    : values_(rows*cols, init_value), reserved_size1_(rows), size1_(rows), size2_(cols)
    {
    }

    template <typename T, typename MemoryBlock>
    matrix<T, MemoryBlock>::matrix(matrix const& m)
    : values_(m.values_), reserved_size1_(m.reserved_size1_), size1_(m.size1_), size2_(m.size2_)
    {
        // TODO improve (if needed)
        // the copy constructor copies the whole data vector including the whole reserved space
        // it could copy only the data that is really in use
        // (if reserved_space/actual_space is above a certain threshold)
    }

    template <typename T, typename MemoryBlock>
    template <typename T2, typename OtherMemoryBlock>
    matrix<T, MemoryBlock>::matrix(matrix<T2,OtherMemoryBlock> const& m)
    :   values_(
              boost::make_transform_iterator( m.values_.begin(), numeric_cast<T,T2> )
            , boost::make_transform_iterator( m.values_.end(),   numeric_cast<T,T2> )
        )
      , reserved_size1_(m.reserved_size1_) , size1_(m.size1_), size2_(m.size2_)
    {
        // TODO improve (if needed)
        // the copy constructor copies the whole data vector including the whole reserved space
        // it could copy only the data that is really in use
        // (if reserved_space/actual_space is above a certain threshold)
    }

    template <typename T, typename MemoryBlock>
    template <typename ForwardIterator>
    matrix<T, MemoryBlock>::matrix(std::vector<std::pair<ForwardIterator,ForwardIterator> > const& columns)
    : values_(), reserved_size1_(0), size1_(0), size2_(0)
    {
        using std::distance;
        using std::copy;
        using std::swap;
        assert(columns.size() > 0);

        size_type const reserve_rows = distance(columns.front().first, columns.front().second);
        MemoryBlock tmp(reserve_rows*columns.size());
        for(std::size_t i=0; i < columns.size(); ++i) {
            assert(distance(columns[i].first,columns[i].second) == static_cast<difference_type>(reserve_rows));
            copy(columns[i].first, columns[i].second, tmp.begin()+reserve_rows*i);
        }
        swap(tmp,values_);
        reserved_size1_ = reserve_rows;
        size1_ = reserve_rows;
        size2_ = columns.size();
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::swap(matrix & r)
    {
        using std::swap;
        swap(this->values_, r.values_);
        swap(this->reserved_size1_,r.reserved_size1_);
        swap(this->size1_, r.size1_);
        swap(this->size2_, r.size2_);
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
        assert(i < this->size1_);
        assert(j < this->size2_);
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
        using std::fill;
        // Do we need more space? Reserve more space if needed!
        automatic_reserve(rows,cols);
        if(rows > this->size1_)
        {
            // Reset all "new" elements which are in already reserved
            // rows of already existing columns to init_value
            size_type const num_of_cols = (std::min)(cols, this->size2_);
            for(size_type j=0; j < num_of_cols; ++j)
                fill(
                      this->values_.begin()+j*this->reserved_size1_ + this->size1_
                    , this->values_.begin()+j*this->reserved_size1_ + rows
                    , init_value
                );
        }
        if(cols > this->size2_)
        {
            // Fill all new cols
            fill(
                  this->values_.begin()+ this->size2_ * this->reserved_size1_
                , this->values_.begin()+ cols * this->reserved_size1_
                , init_value
            );
        }
        this->size1_=rows;
        this->size2_=cols;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::reserve(size_type rows, size_type cols)
    {
        assert( (this->reserved_size1_ == 0 && this->values_.size() == 0 ) || this->values_.size() % this->reserved_size1_ == 0 );
        using std::swap;
        using std::copy;
        // Ignore values that would shrink the matrix
        cols = (std::max)(cols, this->reserved_size1_ == 0 ? 0 : this->values_.size() / this->reserved_size1_);
        rows = (std::max)(rows, this->reserved_size1_);

        // Is change of structure or size of the MemoryBlock necessary?
        if(rows > this->reserved_size1_ || rows*cols > this->values_.size() )
        {
            this->force_reserve(rows,cols);
        }
    }

    template <typename T, typename MemoryBlock>
    std::pair<typename matrix<T,MemoryBlock>::size_type,typename matrix<T,MemoryBlock>::size_type> matrix<T, MemoryBlock>::capacity() const
    {
        assert( (this->reserved_size1_ == 0 && this->values_.size() == 0 ) || this->values_.size() % this->reserved_size1_ == 0 );
        return std::make_pair( this->reserved_size1_, this->reserved_size1_ == 0 ? 0 : values_.size() / this->reserved_size1_ );
    }

    template <typename T, typename MemoryBlock>
    inline bool matrix<T, MemoryBlock>::is_shrinkable() const
    {
        return (this->size1_ < this->reserved_size1_ || this->reserved_size1_*this->size2_ < values_.size() );
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::shrink_to_fit()
    {
        if(this->size1_ == this->reserved_size1_ && this->size1_*this->size2_ == values_.size() )
            return;
        this->force_reserve(this->size1_, this->size2_);
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::clear()
    {
        this->size1_ = 0;
        this->size2_ = 0;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::append_cols(std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        using std::copy;
        using std::distance;
        assert( distance(range.first, range.second) == static_cast<difference_type>(k*this->size1_) );
        // Reserve more space if needed
        automatic_reserve(this->size1_,this->size2_+k);
        // Append column by column
        for(difference_type l=0; l<k; ++l)
            copy(range.first+(l*this->size1_), range.first+((l+1)*this->size1_), values_.begin()+this->reserved_size1_*(this->size2_+l));
        this->size2_ += k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::append_rows(std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        using std::copy;
        using std::distance;
        assert( distance(range.first, range.second) == static_cast<difference_type>(k*this->size2_) );
        // Reserve more space if needed
        automatic_reserve(this->size1_+k, this->size2_);
        // The elements do already exists due to reserve, so we can just use (copy to) them.
        for(difference_type l=0; l<k; ++l)
            copy( range.first+(l*this->size2_), range.first+((l+1)*this->size2_), row_element_iterator(&values_[0]+size1_+l,reserved_size1_) );
        this->size1_ += k;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::insert_rows(size_type i, std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        using std::copy;
        using std::distance;
        assert( i <= this->size1_ );
        assert( distance(range.first, range.second) == static_cast<difference_type>(k*this->size2_) );

        // Append the row
        automatic_reserve(this->size1_+k,this->size2_);


        // Move existing data to make some space for the insertion
        // We can not use std::copy because input and output range overlap
        size_type const newsize1 = this->size1_+k;
        for(size_type j=0; j<this->size2_; ++j)
            for(size_type l=this->size1_; l > i; --l)
                *(this->values_.begin()+this->reserved_size1_*j+k+l-1) = *(this->values_.begin()+this->reserved_size1_*j+l-1);

        // Insert new data
        for(difference_type l=0; l<k; ++l)
            copy(range.first+l*this->size2_,range.first+(l+1)*this->size2_,row_element_iterator(&values_[0]+i+l,reserved_size1_) );
        this->size1_= newsize1;
    }

    template <typename T, typename MemoryBlock>
    template <typename InputIterator>
    void matrix<T, MemoryBlock>::insert_cols(size_type j, std::pair<InputIterator,InputIterator> const& range, difference_type k)
    {
        using std::copy;
        using std::distance;

        assert( j <= this->size2_);
        assert( distance(range.first, range.second) == static_cast<difference_type>(k*this->size1_) );

        // Append the column
        automatic_reserve(this->size1_,this->size2_+k);

        // Move the column through the matrix to the right possition
        for(size_type h=this->size2_; h>j; --h)
            copy(&this->values_[0]+(this->reserved_size1_*(h-1)),&this->values_[0]+(this->reserved_size1_*(h-1)+this->size1_),&this->values_[0]+(this->reserved_size1_*(h+k-1)));
        for(difference_type l=0; l<k; ++l)
            copy(range.first+l*this->size1_,range.first+(l+1)*this->size1_,&this->values_[0]+(this->reserved_size1_*(j+l)));
        this->size2_+=k;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::remove_rows(size_type i, difference_type k)
    {
        using std::copy;
        assert( i+k <= this->size1_ );
        // for each column, copy the rows > i+k   k rows  up
        // We can not use std::copy because input and output range overlap
        size_type const newsize1 = this->size1_-k;
        for(size_type j = 0; j < this->size2_; ++j)
            for(size_type l=i; l < newsize1; ++l)
                *(this->values_.begin()+this->reserved_size1_*j+l) = *(this->values_.begin()+this->reserved_size1_*j+l+k);
        this->size1_ = newsize1;
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::remove_cols(size_type j, difference_type k)
    {
        using std::copy;
        assert( j+k <= this->size2_ );
        for(; j < this->size2_-k; ++j)
            copy(this->values_.begin()+reserved_size1_*(j+k), this->values_.begin()+reserved_size1_*(j+k)+size1_, this->values_.begin()+reserved_size1_*j);
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
    bool matrix<T, MemoryBlock>::automatic_reserve(size_type size1, size_type size2)
    {
        // Do we need to reserve more space in any dimension?
        if(size1 > this->reserved_size1_ || this->reserved_size1_*size2 > this->values_.size())
        {
            this->reserve(size1*3/2,size2*3/2);
            return true;
        }
        else
        {
            return false;
        }
    }

    template <typename T, typename MemoryBlock>
    void matrix<T, MemoryBlock>::force_reserve(size_type rows, size_type cols)
    {
        using std::swap;
        using std::copy;
        // We assume the underlying MemoryBlock doesn't
        // reserve additional space on construction
        MemoryBlock tmp(rows*cols);
        // Copy column by column
        for(size_type j=0; j < this->size2_; ++j)
        {
            std::pair<col_element_iterator, col_element_iterator> range(col(j));
            // Copy the elements from the current MemoryBlock
            copy(range.first, range.second, tmp.begin()+j*rows);
        }
        swap(this->values_,tmp);
        this->reserved_size1_ = rows;
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

//////////////////////////////////////////////////////////////////////////////

    template <typename T, typename MemoryBlock>
    const matrix<T,MemoryBlock> operator - (matrix<T,MemoryBlock> a)
    {
        // Do the operation column by column
        for(typename matrix<T,MemoryBlock>::size_type j=0; j < a.num_cols(); ++j)
        {
#if defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
// Workaround for a compiler bug in clang 3.0 (and maybe earlier versions)
            for(typename matrix<T,MemoryBlock>::size_type i=0; i < a.num_rows(); ++i)
            {
                typename matrix<T,MemoryBlock>::value_type const tmp = -a(i,j);
                a(i,j) = tmp;
            }
#else // defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
            std::pair<typename matrix<T,MemoryBlock>::col_element_iterator,
                typename matrix<T,MemoryBlock>::col_element_iterator> range(a.col(j));
            std::transform(range.first, range.second,range.first, std::negate<T>());
#endif // defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
        }
        return a;
    }

    template<class T, class MemoryBlock>
    std::size_t size_of(matrix<T, MemoryBlock> const & m)
    {
        return num_rows(m)*num_cols(m)*sizeof(T);
    }

    template <typename T, typename MemoryBlock>
    std::ostream& operator << (std::ostream& os, matrix<T,MemoryBlock> const& m)
    {
        detail::print_matrix(os, m);
        return os;
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
