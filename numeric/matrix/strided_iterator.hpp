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

#ifndef __ALPS_STRIDED_ITERATOR_HPP__
#define __ALPS_STRIDED_ITERATOR_HPP__

#include <boost/iterator/iterator_facade.hpp>
#include <boost/static_assert.hpp>
#include <cassert>

namespace alps {
namespace numeric {

template <typename Matrix, typename T>
class strided_iterator : public boost::iterator_facade<
                                strided_iterator<Matrix,T>,
                                T,
                                boost::random_access_traversal_tag,
                                T&,
                                typename Matrix::difference_type
                                >
{
    public:
        typedef T value_type;

        strided_iterator(value_type* ptr, typename Matrix::difference_type stride)
            : ptr(ptr), stride(stride)
        {
            // The value_type of the iterator must be the value_type of the matrix or const Matrix::value_type
            BOOST_STATIC_ASSERT( (boost::is_same<typename Matrix::value_type, T>::value
                                 || boost::is_same<const typename Matrix::value_type,T>::value) );
        }

        template<typename Matrix2, typename U>
        strided_iterator(strided_iterator<Matrix2,U> const& r)
            : ptr(r.ptr), stride(r.stride)
        {
            BOOST_STATIC_ASSERT( (boost::is_same<
                        typename boost::add_const<Matrix>::type,
                        typename boost::add_const<Matrix2>::type
                        >::value ));
        }

    private:
        friend class boost::iterator_core_access;
        template <typename,typename> friend class strided_iterator;

        value_type& dereference() const
        { return *ptr; }

        template <typename Matrix2,typename U>
        bool equal(strided_iterator<Matrix2,U> const& y) const
        {
            BOOST_STATIC_ASSERT( (boost::is_same<
                        typename boost::add_const<Matrix>::type,
                        typename boost::add_const<Matrix2>::type
                        >::value ));
            if(ptr == y.ptr)
                return true;
            else
                return false;
        }
        void increment()
        {
            ptr+=stride;
        }
        void decrement()
        {
            ptr-=stride;
        }
        void advance(typename Matrix::difference_type n)
        {
            ptr += n*stride;
        }

        template <typename U>
        typename Matrix::difference_type distance_to(strided_iterator<Matrix,U> const& z) const
        {
            assert( (z.ptr - ptr) % stride == 0 );
            return (z.ptr - ptr)/stride;
        }

        value_type* ptr;
        typename Matrix::difference_type stride; 
};

} // end namespace numeric
} // end namespace alps

#endif //__ALPS_STRIDED_ITERATOR_HPP__
