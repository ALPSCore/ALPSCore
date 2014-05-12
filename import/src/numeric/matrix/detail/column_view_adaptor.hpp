/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2010 - 2013 by Andreas Hehn <hehn@phys.ethz.ch>                   *
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
#ifndef ALPS_NUMERIC_MATRIX_COLUMN_VIEW_ADAPTOR_HPP
#define ALPS_NUMERIC_MATRIX_COLUMN_VIEW_ADAPTOR_HPP

#include <boost/numeric/bindings/detail/adaptor.hpp>

namespace alps { namespace numeric {
    template <typename Matrix>
    class column_view;
} }

//
// An adaptor for the column_view of alps::numeric::matrix to the boost::numeric::bindings
//

namespace boost { namespace numeric { namespace bindings { namespace detail {

    template< typename T, typename MemoryBlock, typename Id, typename Enable >
    struct adaptor< ::alps::numeric::column_view< ::alps::numeric::matrix<T,MemoryBlock> >, Id, Enable>
    {
        typedef typename copy_const< Id, T >::type value_type;
        typedef std::ptrdiff_t  size_type;

        typedef mpl::map<
            mpl::pair< tag::value_type,     value_type >,
            mpl::pair< tag::entity,         tag::vector >,
            mpl::pair< tag::size_type<1>,   size_type >,
            mpl::pair< tag::data_structure, tag::linear_array >,
            mpl::pair< tag::stride_type<1>, tag::contiguous >
        > property_map;

        static std::ptrdiff_t size1( const Id& id ) {
            return id.size();
        }

        static value_type* begin_value( Id& id ) {
            return &(*id.begin());
        }

        static value_type* end_value( Id& id ) {
            return &(*id.end());
        }
    };
}}}}

#endif // ALPS_NUMERIC_MATRIX_COLUMN_VIEW_ADAPTOR_HPP
