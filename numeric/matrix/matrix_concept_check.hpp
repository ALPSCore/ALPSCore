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

#ifndef ALPS_MATRIX_CONCEPT_CHECK_HPP
#define ALPS_MATRIX_CONCEPT_CHECK_HPP
#include <boost/concept_check.hpp>
#include <boost/type_traits/remove_const.hpp>

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
        
        // TODO write more restrictive BOOST_CONCEPT_ASSERTs for iterators
        typedef typename X::row_element_iterator                row_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<row_element_iterator>));
        typedef typename X::const_row_element_iterator          const_row_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<const_row_element_iterator>));
        typedef typename X::col_element_iterator             col_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<col_element_iterator>));
        typedef typename X::const_col_element_iterator       const_col_element_iterator;
        BOOST_CONCEPT_ASSERT((boost::InputIterator<const_col_element_iterator>));
        

    BOOST_CONCEPT_USAGE(Matrix)
    {
        // Constructor
        typename boost::remove_const<X>::type x(1,1);
        // Copy constructor
        const X y(x);
        typename boost::remove_const<X>::type z = x;

        // Swap
        std::swap(x,z);

        // num_rows(), num_cols()
        std::size_t s = num_rows(y);
        s = num_cols(y);
        
        // Element access
        t = x(0,0);
        x(0,0)+=value_type();

        // Swap rows/cols
//        swap_rows(x,0,1);
//        swap_cols(x,0,1);
        
        // Iterator functions
        std::pair<row_element_iterator,row_element_iterator>                    row_range = row(x,0);
        std::pair<col_element_iterator,col_element_iterator>                    col_range = col(x,0);
        std::pair<const_row_element_iterator,const_row_element_iterator>        const_row_range = row(y,0);
        std::pair<const_col_element_iterator,const_col_element_iterator>        const_col_range = col(y,0);

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

} // namespace numeric
} // namespace alps

#endif //ALPS_MATRIX_CONCEPT_CHECK_HPP
