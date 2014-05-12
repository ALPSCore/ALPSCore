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

#ifndef ALPS_MATRIX_ADAPTOR_HPP
#define ALPS_MATRIX_ADAPTOR_HPP

#include <boost/numeric/bindings/detail/adaptor.hpp>

namespace alps { namespace numeric {
    template <typename T, typename MemoryBlock>
    class matrix;
} }

//
// An adaptor for the matrix to the boost::numeric::bindings
//

namespace boost { namespace numeric { namespace bindings { namespace detail {

    template <typename T, typename MemoryBlock, typename Id, typename Enable>
    struct adaptor< ::alps::numeric::matrix<T,MemoryBlock>, Id, Enable>
    {
        typedef typename copy_const< Id, T >::type              value_type;
        // TODO: fix the types of size and stride -> currently it's a workaround, since std::size_t causes problems with boost::numeric::bindings
        //typedef typename ::alps::numeric::matrix<T,Alloc>::size_type         size_type;
        //typedef typename ::alps::numeric::matrix<T,Alloc>::difference_type   difference_type;
        typedef std::ptrdiff_t  size_type;
        typedef std::ptrdiff_t  difference_type;

        typedef mpl::map<
            mpl::pair< tag::value_type,      value_type >,
            mpl::pair< tag::entity,          tag::matrix >,
            mpl::pair< tag::size_type<1>,    size_type >,
            mpl::pair< tag::size_type<2>,    size_type >,
            mpl::pair< tag::data_structure,  tag::linear_array >,
            mpl::pair< tag::data_order,      tag::column_major >,
            mpl::pair< tag::data_side,       tag::upper >,
            mpl::pair< tag::stride_type<1>,  tag::contiguous >,
            mpl::pair< tag::stride_type<2>,  difference_type >
        > property_map;

        static size_type size1( const Id& id ) {
            return id.num_rows();
        }

        static size_type size2( const Id& id ) {
            return id.num_cols();
        }

        static value_type* begin_value( Id& id ) {
            return &(*id.col(0).first);
        }

        static value_type* end_value( Id& id ) {
            return &(*(id.col(id.num_cols()-1).second-1));
        }

        static difference_type stride1( const Id& id ) {
            return id.stride1();
        }

        static difference_type stride2( const Id& id ) {
           return id.stride2();
        }

    };

}}}}

#endif //ALPS_MATRIX_ADAPTOR_HPP
