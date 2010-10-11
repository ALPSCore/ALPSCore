#ifndef __ALPS_GENERAL_MATRIX_HPP__
#define __ALPS_GENERAL_MATRIX_HPP__

#include <vector>
#include <algorithm>
#include <functional>

namespace blas {

    template <typename T>
    class general_matrix {
    public:
        typedef T value_type;
        //TODO: alignment!
        general_matrix(std::size_t size1, std::size_t size2, T init_value = T(0) )
        : size1_(size1), size2_(size2), reserved_size1_(size1), values_(size1*size2, init_value)
        {
        }

        general_matrix(std::size_t size=0, T init_value = T(0) )
        : size1_(size), size2_(size), reserved_size1_(size), values_(size*size, T(0))
        {
        }    
        
        general_matrix(const general_matrix& mat)
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
                for(unsigned int j=0; j < mat.size2_; ++j)
                    std::copy( mat.values_.begin()+j*mat.reserved_size1_, mat.values_.begin()+j*mat.reserved_size1_+mat.size1_, values_.begin() );
            }
        }

        friend void swap(general_matrix& x,general_matrix& y)
        {
            std::swap(x.values_, y.values_);
            std::swap(x.size1_, y.size1_);
            std::swap(x.size2_, y.size2_);
            std::swap(x.reserved_size1_,y.reserved_size1_);
        }
        
        general_matrix& operator = (general_matrix rhs)
        {
            swap(rhs, *this);
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

        inline T &operator()(const unsigned i, const unsigned j)
        {
            assert((i < size1_) && (j < size2_));
            return values_[j*reserved_size1_+i];
        }
        
        inline const T &operator()(const unsigned i, const unsigned j) const 
        {
            assert((i < size1_) && (j < size2_));
            return values_[j*reserved_size1_+i];
        }

        inline const std::size_t num_rows() const
        {
            return size1_;
        }

        inline const std::size_t num_cols() const
        {
            return size2_;
        }
        
        //TODO: shall these functions be kept for compatibility or can we drop them?
        inline const std::size_t size1() const 
        {
            return size1_;
        }
  
        inline const std::size_t size2() const
        { 
            return size2_;
        }
       
        // SQUARE MATRIX These functions make only sense for square matrices

        // free function!!!
        void set_to_identity()
        {
            assert(size1_==size2_);
            clear();
            for(unsigned int i=0;i<size1_;++i)
            {
                operator()(i,i)=T(1);
            }
        }
        // SQUARE MATRIX END

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
                    for(unsigned int j=0; j < size2_; ++j)
                    {
                        std::fill(values_.begin()+j*reserved_size1_ + size1_, values_.begin()+j*reserved_size1_ + size1, init_value);
                    }
                }

            }
            else // size1 > reserved_size1_
            {
                std::vector<T> tmp(size1*size2,init_value);
                for(unsigned int j=0; j < size2_; ++j)
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
                for(unsigned int j=0; j < size2_; ++j)
                {
                    // Copy column by column
                    std::copy( values_.begin()+j*reserved_size1_, values_.begin()+j*reserved_size1_+size1_, tmp.begin()+j*size1);
                }
                std::swap(values_,tmp);
                reserved_size1_ = size1;
            }
        }

        std::pair<std::size_t,std::size_t> capacity()
        {
            assert( values_.capacity() % reserved_size1_ == 0 );
            // Evaluate the maximal number of columns (with size reserved_size1_) that the underlying vector could hold.
            // If the constructor, resize() and reserve() of std::vector would guarantee to allocate 
            // the requested amount of memory exactly
            // values_.capacity() % reserved_size1_ == 0 should hold.
            // However these functions guarantee to allocate _at least_ the requested amount.
            // TODO: perhaps there is a better implementation
            std::size_t reserved_size2_ = values_.capacity() - (values_.capacity() % reserved_size1_) / reserved_size1_;
            return std::pair<std::size_t,std::size_t>( reserved_size1_, reserved_size2_ );
        }
        
        void clear()
        {
            // Clear the values vector and ensure the reserved size stays the way it was
            values_.clear();
            values_.resize(reserved_size1_*size2_);
            size1_ = 0;
            size2_ = 0;
        }

    // hooks, and make it a free function

        vector<T> get_column(unsigned int j) const
        {
            assert(j < size2_);
            return vector<T>(values_.begin()+j*reserved_size1_,values_.begin()+j*reserved_size1_+size1_);
        }

        vector<T> get_row(unsigned int i) const
        {
            assert(i < size1_);
            vector<T> result(size2_);
            for(unsigned int j=0; j < size2_; ++j)
            {
                result(j) = operator()(i,j);
            }
            return result;
        }

        void set_column(unsigned int j, vector<T> const& v)
        {
            assert( j < size2_ );
            assert( v.size() == size1_ );
            std::copy(v.begin(),v.end(),values_.begin()+j*reserved_size1_);
        }

        void set_row(unsigned int i, vector<T> const& v)
        {
            assert( i < size1_ );
            assert( v.size() == size2_ );
            // TODO better implementation
            for(unsigned int j=0; j < size1_; ++j)
            {
                operator()(i,j) = v(j);
            }
        }

        void append_column(vector<T> const& v)
        {
            assert( v.size() == size1_ );
            unsigned int insert_position = size2_;
            resize(size1_,size2_+1);
            std::copy(v.begin(),v.end(),values_.begin()+reserved_size1_*size2_);
        }

        void apped_row(vector<T> const& v)
        {
            assert( v.size() == size2_ );
            unsigned int insert_position = size1_;
            resize(size1_+1,size2_);
            for(unsigned int j=0; j < size2_; ++j)
            {
                operator()(insert_position,j) = v(j);
            }
        }

        void insert_row(unsigned int i, vector<T> const& v)
        {
            assert( i <= size1_ );
            assert( v.size() == size2_ );

            // Append the row
            append_row(v);

            // Move the row through the matrix to the right possition
            for(unsigned int k=size1_-1; k>i; ++k)
            {
                swap_rows(k,k-1);
            }
        }

        void insert_column(unsigned int j, vector<T> const& v)
        {
            assert( j <= size2_);
            assert( v.size() == size1_ );
            
            // Append the column
            append_column(v);

            // Move the column through the matrix to the right possition
            for(unsigned int k=size2_-1; k>j; ++k)
            {
                swap_columns(k,k-1);
            }

        }

        void swap_columns(unsigned int j1, unsigned int j2)
        {
            assert( j1 < size2_ && j2 < size2_ );
            std::swap_ranges(values_.begin()+j1*reserved_size1_,values_.begin()+j1*reserved_size1_+size1_, values_.begin()+j2*reserved_size1_);
        }
        
        void swap_rows(unsigned int i1, unsigned int i2)
        {
            assert( i1 < size1_ && i2 < size1_ );
            // TODO find a better implementation
            for(unsigned int j=0; j < size2_; ++j)
            {
                std::swap(values_[j*reserved_size1_+i1], values_[j*reserved_size1_+i2]);
            }
        }

        inline const T trace() const
        {
            assert(size1_==size2_);
            T tr= T(0);
            for(unsigned int i=0; i<size1_; ++i) tr+=operator()(i,i);
            return tr;
        }
        
        void transpose() 
        {
            // TODO: perhaps this could be reimplemented as a free function returning a proxy object
            if(size1_==0 || size2_==0) return;
            general_matrix tmp(size2_,size1_);
            for(unsigned int i=0;i<size1_;++i){
                for(unsigned int j=0;j<size2_;++j){
                    tmp(j,i) = operator()(i,j);
                }
            }
            std::swap(tmp,*this);
        }

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
        
        general_matrix<T>& operator += (const general_matrix &rhs) 
        {
            assert((rhs.size1() == size1_) && (rhs.size2() == size2_));
            if(!(this->is_shrinkable() || rhs->is_shrinkable()) )
            {
                std::transform(values_.begin(),values_.end(),rhs.values_.begin(),values_.begin(), std::plus<T>());
            }
            else
            {
                // Do the operation column by column
                for(unsigned int j=0; j < size2_; ++j)
                {
                    std::transform(values_.begin()+j*reserved_size1_, values_.begin()+j*reserved_size1_+size1_, rhs.values_.begin()+j*rhs.reserved_size1_, values_.begin()+j*reserved_size1_, std::plus<T>());
                }
            }
            return *this;
        }
        
        general_matrix<T>& operator -= (const general_matrix &rhs) 
        {
            assert((rhs.size1() == size1_) && (rhs.size2() == size2_));
            if(!(this->is_shrinkable() || rhs->is_shrinkable()) )
            {
                std::transform(values_.begin(),values_.end(),rhs.values_.begin(),values_.begin(), std::minus<T>());
            }
            else
            {
                // Do the operation column by column
                for(unsigned int j=0; j < size2_; ++j)
                {
                    std::transform(values_.begin()+j*reserved_size1_, values_.begin()+j*reserved_size1_+size1_, rhs.values_.begin()+j*rhs.reserved_size1_, values_.begin()+j*reserved_size1_, std::minus<T>());
                }
            }
            return *this;
        }
        

        general_matrix<T>& operator *= (T const& t)
        {
            if(!(this->is_shrinkable()) )
            {
                std::transform(values_.begin(),values_.end(),values_.begin(), binder1st(std::multiplies<T>(),t));
            }
            else
            {
                // Do the operation column by column
                for(unsigned int j=0; j < size2_; ++j)
                {
                    std::transform(values_.begin()+j*reserved_size1_, values_.begin()+j*reserved_size1_+size1_, rhs.values_.begin()+j*rhs.reserved_size1_, values_.begin()+j*reserved_size1_, binder1st(std::multiplies<T>(),t));
                }
            }
            return *this;
        }
        
        general_matrix<T>& operator *= (const general_matrix &rhs) 
        {
            assert( size2_ == rhs.size1() );

            // Simple matrix matrix multiplication
            general_matrix<T> tmp(size1_,rhs.size2());
            for(unsigned int i=0; i < size1_; ++i)
            {
                for(unsigned int j=0; j<rhs.size2(); ++j)
                {
                    for(unsigned int k=0; k<size2_; ++k)
                    {
                            tmp(i,j) += operator()(i,k) * rhs(k,j);
                    }
                }
            }
            std::swap(tmp,*this);
            return *this;
        }

    private:
        bool is_shrinkable()
        {
            // This assertion should actually never fail
            assert( reserved_size1_*size2_ == values_.size() );
            if(size1_ < reserved_size1_) return true;
            else return false;
        }


        std::size_t size1_;
        std::size_t size2_;
        std::size_t reserved_size1_;
        // "reserved_size2_" is done automatically by underlying std::vector (see vector.reserve(), vector.capacity() )
        
        std::vector<T> values_;
    };

    template<typename T>
    const vector<T> operator * (general_matrix<T> const& m, vector<T> const& v)
    {
        assert( m.size2() == v.size() );
        vector<T> result(m.size1());
        // Simple Matrix * Vector
        for(unsigned int i = 0; i < m.size1(); ++i)
        {
            for(unsigned int j=0; j <m.size2(); ++j)
            {
                result(i) = m(i,j) * v(j);
            }
        }
        return result;
    }
   
    // TODO: adj(Vector) * Matrix, where adj is a proxy object

    template<typename T>
    const general_matrix<T> operator * (general_matrix<T> m1, general_matrix<T> const& m2)
    {
        m1 *= m2;
        return m1;
    }
    

    //TODO Check whether this compiles or there are problems with T being a general_matrix
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
} // namespace blas

#endif //__ALPS_GENERAL_MATRIX_HPP__
