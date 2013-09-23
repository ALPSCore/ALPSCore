/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
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
#ifndef ALPS_NUMERIC_MATRIX_SCALAR_PRODUCT_HPP
#define ALPS_NUMERIC_MATRIX_SCALAR_PRODUCT_HPP

#include <alps/numeric/matrix/detail/debug_output.hpp>
#include <alps/numeric/matrix/entity.hpp>
#include <alps/numeric/matrix/is_blas_dispatchable.hpp>
#include <alps/numeric/matrix/detail/auto_deduce_multiply_return_type.hpp>
#include <alps/functional.h>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/numeric/bindings/blas/level1/dot.hpp>
#include <functional>
#include <numeric>


namespace alps {
namespace numeric {


template <typename T1, typename T2>
struct value_type_scalar_product_return_type
{
    typedef typename detail::auto_deduce_multiply_return_type<typename T1::value_type,typename T2::value_type>::type type;
};

template <typename T1, typename T2>
struct scalar_product_return_type
{
    typedef /*typename  boost::enable_if<
          typename boost::mpl::and_<
                boost::is_same<typename get_entity<T1>::type, tag::vector>
              , boost::is_same<typename get_entity<T2>::type, tag::vector>
          >::type
        , */
	typename value_type_scalar_product_return_type<T1,T2>::type
    //>::type 
	type;
};


template <typename T1, typename T2>
typename scalar_product_return_type<T1,T2>::type scalar_product_impl(T1 const& t1, T2 const& t2, boost::mpl::false_)
{
    return std::inner_product(t1.begin(), t1.end(), t2.begin()
        , typename scalar_product_return_type<T1,T2>::type()
        , std::plus<typename scalar_product_return_type<T1,T2>::type>()
        , conj_mult<typename T1::value_type, typename T2::value_type>()
    );
}

template <typename T1, typename T2>
typename scalar_product_return_type<T1,T2>::type scalar_product_impl(T1 const& t1, T2 const& t2, boost::mpl::true_)
{
    ALPS_NUMERIC_MATRIX_DEBUG_OUTPUT( "using blas dot for " << typeid(t1).name() << " " << typeid(t2).name() );
    return boost::numeric::bindings::blas::dot(t1,t2);
}

template <typename T1, typename T2>
typename scalar_product_return_type<T1,T2>::type scalar_product(T1 const& t1, T2 const& t2)
{
    assert(t1.size() == t2.size());
    return scalar_product_impl(t1,t2,is_blas_dispatchable<T1,T2>());
}

}
}
#endif // ALPS_NUMERIC_MATRIX_SCALAR_PRODUCT_HPP
