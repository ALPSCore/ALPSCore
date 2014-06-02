/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MATRIX_CONCEPT_CHECK_HPP
#define ALPS_MATRIX_CONCEPT_CHECK_HPP

// To get rid of annoying unused value warnings.
#define ALPS_UNUSED(x) \
    static_cast<void>(x)

#include <boost/concept_check.hpp>
#include <boost/type_traits/remove_const.hpp>
#include <boost/type_traits/add_const.hpp>

namespace alps {
namespace numeric {

template <typename X>
struct Matrix : boost::Assignable<X>
{
    public:
        typedef typename X::value_type                          value_type;
        // TODO BOOST_CONCEPT_ASSERT((Field<value_type>));
        typedef typename X::size_type                           size_type;
        BOOST_CONCEPT_ASSERT((boost::UnsignedInteger<size_type>));
        typedef typename X::difference_type                     difference_type;
        BOOST_CONCEPT_ASSERT((boost::SignedInteger<difference_type>));



    BOOST_CONCEPT_USAGE(Matrix)
    {
        // Constructor
        typename boost::remove_const<X>::type x(1,1);
        // Copy constructor
        const X y(x);
        typename boost::remove_const<X>::type z = x;

        // Swap
        swap(x,z);

        // num_rows(), num_cols()
        std::size_t s = num_rows(y);
        s = num_cols(y);
        ALPS_UNUSED(s);

        // Element access
        t = x(0,0);
        x(0,0)+=value_type();

        // Swap rows/cols
//        swap_rows(x,0,1);
//        swap_cols(x,0,1);


        // operators
        z = x;
        x += y;
        x -= y;
        x *= t;

        z = x + y;
        z = x - y;
        z = x * y;

        // Matrix vector multiplication
        // this does not check for mixed types
//#warning FIXME
        /* Which vector class is this supposed to use in the general case? */
//        vector<value_type> v;
//        v = x * v;

    }

    private:
        // Default constructable value_type
        value_type t;
};


template <typename X>
struct DiagonalIteratableMatrix : Matrix<X>
{
    public:
        // TODO write more restrictive BOOST_CONCEPT_ASSERTs for iterators
        typedef typename X::diagonal_iterator                diagonal_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<diagonal_iterator>));
        typedef typename X::const_diagonal_iterator          const_diagonal_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<const_diagonal_iterator>));
    BOOST_CONCEPT_USAGE(DiagonalIteratableMatrix)
    {
        typename boost::remove_const<X>::type x(1,1);
        typename boost::add_const<X>::type y(1,1);

        std::pair<diagonal_iterator,diagonal_iterator>              diagonal_range = diagonal(x);
        std::pair<const_diagonal_iterator,const_diagonal_iterator>  const_diagonal_range = diagonal(y);
        ALPS_UNUSED(diagonal_range);
        ALPS_UNUSED(const_diagonal_range);
    }
};

template <typename X>
struct IteratableMatrix : DiagonalIteratableMatrix<X>
{
    public:
        // TODO write more restrictive BOOST_CONCEPT_ASSERTs for iterators
        typedef typename X::row_element_iterator                row_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<row_element_iterator>));
        typedef typename X::const_row_element_iterator          const_row_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<const_row_element_iterator>));
        typedef typename X::col_element_iterator             col_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<col_element_iterator>));
        typedef typename X::const_col_element_iterator       const_col_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<const_col_element_iterator>));

    BOOST_CONCEPT_USAGE(IteratableMatrix)
    {
        typename boost::remove_const<X>::type x(1,1);
        typename boost::add_const<X>::type y(1,1);
        // Iterator functions
        std::pair<row_element_iterator,row_element_iterator>                    row_range = row(x,0);
        std::pair<col_element_iterator,col_element_iterator>                    col_range = col(x,0);
        std::pair<const_row_element_iterator,const_row_element_iterator>        const_row_range = row(y,0);
        std::pair<const_col_element_iterator,const_col_element_iterator>        const_col_range = col(y,0);
    }
};

} // namespace numeric
} // namespace alps

#undef ALPS_UNUSED
#endif //ALPS_MATRIX_CONCEPT_CHECK_HPP
