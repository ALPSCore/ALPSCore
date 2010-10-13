
template <typename matrix_type>
class general_iterator
{
    public:
        typedef typename matrix_type::value_type value_type;
        general_iterator()
            : row_pos(0), col_pos(0)
            {}
        general_iterator(general_iterator const& r)
            :row_pos(r.row_pos),col_pos(r.col_pos)
            {}

        general_iterator operator++(int i)
        {
            general_iterator tmp(*this);
            ++(*this);
            return tmp;
        }
        value_type operator*()
        {
            return m(row_pos,col_pos);
        }
    private:
        std::size_t row_pos;
        std::size_t col_pos;
};

template <typename matrix_type>
class row_iterator : public general_iterator<matrix_type>
{
    // iterates over matrix elements within the same row
    general_iterator& operator++()
    {
        ++row_pos;
    };
};

template <typename matrix_type>
class column_iterator : public general_iterator<matrix_type>
{
    // iterates over matrix elements within the same column
    general_iterator& operator++()
    {
        ++col_pos;
    };
};

class element_iterator
{
    // iterates over matrix elements independent of row and column
};
