#ifndef BLAS_MATRIX_ITERATORS
#define BLAS_MATRIX_ITERATORS

#include <boost/iterator/iterator_facade.hpp>
#include <boost/static_assert.hpp>

template <typename matrix_type, typename value_t>
class matrix_column_iterator;

template <typename matrix_type, typename value_t>
class matrix_row_iterator : public boost::iterator_facade<
                                matrix_row_iterator<matrix_type,value_t>,
                                value_t,
                                boost::random_access_traversal_tag,
                                value_t&,
                                typename matrix_type::difference_type
                                >
{
    // iterates over matrix elements within the same column


    public:
        typedef value_t value_type;

        matrix_row_iterator(matrix_type* matrix,typename matrix_type::size_type row, typename matrix_type::size_type col)
            : m(matrix), row_pos(row), col_pos(col)
        {
            // The value_type of the iterator must be the value_type of the matrix or const matrix_type::value_type
            BOOST_STATIC_ASSERT( (boost::is_same<typename matrix_type::value_type, value_t>::value
                                 || boost::is_same<const typename matrix_type::value_type,value_t>::value) );
        }

        template<typename other_matrix_type, typename other_value_type>
        matrix_row_iterator(matrix_row_iterator<other_matrix_type,other_value_type> const& r)
            : m(r.m), row_pos(r.row_pos), col_pos(r.col_pos)
            {}

        template<typename other_matrix_type, typename other_value_type>
        explicit matrix_row_iterator(matrix_column_iterator<other_matrix_type,other_value_type> const& col_iter)
            : m(col_iter.matrix), row_pos(col_iter.row), col_pos(col_iter.col)
            {}

    private:
        friend class boost::iterator_core_access;
        template <typename,typename> friend class matrix_row_iterator;

        value_type& dereference() const
        { return m->operator()(row_pos,col_pos); }

        // iterators are equal if they point to the same row of the same matrix
        // WARNING: since the column position is not compared
        // two iterators can be equal although they point to different elements
        template <typename other_value_type>
        bool equal(matrix_row_iterator<matrix_type,other_value_type> const& y) const
        {
            if(m == y.m && row_pos == y.row_pos)
                return true;
            else
                return false;
        }
        void increment()
        {
            ++(this->row_pos);
        }
        void decrement()
        {
            --(this->row_pos);
        }
        void advance(typename matrix_type::difference_type n)
        {
            (this->row_pos) += n;
        }

        template <typename other_value_type>
        typename matrix_type::difference_type distance_to(matrix_row_iterator<matrix_type,other_value_type> const& z) const
        {
            return z.row_pos - row_pos;
        }



        typename matrix_type::size_type row_pos;
        typename matrix_type::size_type col_pos;
        matrix_type* m;
};

template <typename matrix_type, typename value_t>
class matrix_column_iterator : public boost::iterator_facade<
                                   matrix_column_iterator<matrix_type,value_t>,
                                   value_t,
                                   boost::random_access_traversal_tag,
                                   value_t&,
                                   typename matrix_type::difference_type
                                   >
{
    // iterates over matrix elements within the same row
    

    public:
        typedef value_t value_type;

        matrix_column_iterator(matrix_type* matrix,typename matrix_type::size_type row, typename matrix_type::size_type col)
            : m(matrix), row_pos(row), col_pos(col)
            {
                // The value_type of the iterator must be the value_type of the matrix or const matrix_type::value_type
                BOOST_STATIC_ASSERT( (boost::is_same<typename matrix_type::value_type, value_t>::value
                                     || boost::is_same<const typename matrix_type::value_type,value_t>::value) );
            }

        template<typename other_matrix_type, typename other_value_type>
        explicit matrix_column_iterator(matrix_row_iterator<other_matrix_type,other_value_type> const& row_iter)
            : m(row_iter.matrix), row_pos(row_iter.row_pos), col_pos(row_iter.col_pos)
            {}
        
        template<typename other_matrix_type, typename other_value_type>
        matrix_column_iterator(matrix_column_iterator<other_matrix_type,other_value_type> const& r)
            : m(r.matrix), row_pos(r.row_pos), col_pos(r.col_pos)
            {}
    
    private:
        friend class boost::iterator_core_access;
        template <typename,typename> friend class matrix_row_iterator;

        value_type& dereference() const
        { return m->operator()(row_pos,col_pos); }

        // see comment for matrix_row_iterator::equal() and swap "row", "column"
        template <typename other_value_type>
        bool equal(matrix_column_iterator<matrix_type,other_value_type> const& y) const
        {
            if(m == y.m && col_pos == y.col_pos)
                return true;
            else
                return false;
        }
        void increment()
        {
            ++(this->col_pos);
        }
        void decrement()
        {
            --(this->col_pos);
        }
        void advance(typename matrix_type::difference_type n)
        {
            (this->col_pos) += n;
        }

        template <typename other_value_type>
        typename matrix_type::difference_type distance_to(matrix_column_iterator<matrix_type,other_value_type> const& z) const
        {
            return z.col_pos - col_pos;
        }
        
        typename matrix_type::size_type row_pos;
        typename matrix_type::size_type col_pos;
        matrix_type* m;
};

class matrix_element_iterator
{
    // iterates over matrix elements independent of row and column
};

#endif //BLAS_MATRIX_ITERATORS
