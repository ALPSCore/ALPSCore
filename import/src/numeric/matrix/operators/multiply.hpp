/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_MATRIX_OPERATORS_MULTIPLY_HPP
#define ALPS_NUMERIC_MATRIX_OPERATORS_MULTIPLY_HPP

#include <alps/numeric/matrix/detail/auto_deduce_multiply_return_type.hpp>
#include <alps/numeric/matrix/entity.hpp>
#include <alps/numeric/matrix/exchange_value_type.hpp>
#include <boost/type_traits/remove_const.hpp>

namespace alps {
namespace numeric {

template <typename T1, typename T2, typename EntityTag1, typename EntityTag2>
struct multiply_return_type
{
};

template <typename T1, typename T2>
struct multiply_return_type<T1,T2,tag::matrix,tag::matrix>
{
};

template <typename T1, typename T2>
struct multiply_return_type<T1,T2,tag::matrix,tag::vector>
{
    typedef typename exchange_value_type<T2,typename detail::auto_deduce_multiply_return_type<typename T1::value_type,typename T2::value_type>::type>::type type;
};

template <typename T1, typename T2>
struct multiply_return_type<T1,T2,tag::matrix,tag::scalar>
{
    typedef T1 type;
};

template <typename T1, typename T2>
struct multiply_return_type<T1,T2,tag::scalar,tag::matrix>
: multiply_return_type<T2,T1,tag::matrix,tag::scalar>
{
};

template <typename T1, typename T2>
struct multiply_return_type<T1,T2,tag::vector,tag::scalar>
{
    typedef T1 type;
};

template <typename T1, typename T2>
struct multiply_return_type<T1,T2,tag::scalar,tag::vector>
: multiply_return_type<T2,T1,tag::vector,tag::scalar>
{
};

template <typename T1, typename T2>
struct multiply_return_type_helper
: multiply_return_type<
      typename boost::remove_const<T1>::type
    , typename boost::remove_const<T2>::type
    , typename get_entity<T1>::type
    , typename get_entity<T2>::type
> {
};


template <typename T1, typename T2>
typename multiply_return_type_helper<T1,T2>::type operator * (T1 const& t1, T2 const& t2)
{
    return multiply(t1, t2, typename get_entity<T1>::type(), typename get_entity<T2>::type());
}

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_MATRIX_OPERATORS_MULTIPLY_HPP
