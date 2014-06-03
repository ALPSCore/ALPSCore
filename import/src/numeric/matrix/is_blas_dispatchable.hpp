/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#ifndef ALPS_NUMERIC_IS_BLAS_DISPATCHABLE_HPP
#define ALPS_NUMERIC_IS_BLAS_DISPATCHABLE_HPP

#include <boost/type_traits/remove_const.hpp>
#include <boost/mpl/bool_fwd.hpp>
#include <boost/mpl/and.hpp>
#include <alps/numeric/matrix/detail/blasmacros.hpp>

namespace alps {
namespace numeric {

template <typename T>
struct supports_blas : boost::mpl::false_
{
};

#define ALPS_FUNDAMENTAL_TYPES_BLAS_TRAITS(T) \
template <> \
struct supports_blas<T> : boost::mpl::true_ {};

ALPS_IMPLEMENT_FOR_ALL_BLAS_TYPES(ALPS_FUNDAMENTAL_TYPES_BLAS_TRAITS)

#undef ALPS_FUNDAMENTAL_TYPES_BLAS_TRAITS

template <typename T1, typename T2>
struct is_blas_dispatchable;

template <typename T1, typename T2, typename Tag1, typename Tag2>
struct is_blas_dispatchable_helper
:   boost::mpl::and_<
          typename boost::mpl::and_<
                supports_blas<typename boost::remove_const<T1>::type>
              , supports_blas<typename boost::remove_const<T2>::type>
          >::type
        , is_blas_dispatchable<typename T1::value_type, typename T2::value_type>
    >::type
{
};

template <typename T1, typename T2, typename Tag1>
struct is_blas_dispatchable_helper<T1,T2,Tag1,tag::scalar>
:   boost::mpl::and_<
          typename boost::mpl::and_<
                supports_blas<typename boost::remove_const<T1>::type>
              , supports_blas<typename boost::remove_const<T2>::type>
          >::type
        , is_blas_dispatchable<typename T1::value_type, T2>
    >::type
{
};

template <typename T1, typename T2, typename Tag2>
struct is_blas_dispatchable_helper<T1,T2,tag::scalar,Tag2>
:   boost::mpl::and_<
          typename boost::mpl::and_<
                supports_blas<typename boost::remove_const<T1>::type>
              , supports_blas<typename boost::remove_const<T2>::type>
          >::type
        , is_blas_dispatchable<T1, typename T2::value_type>
    >::type
{
};

template <typename T1, typename T2>
struct is_blas_dispatchable_helper<T1,T2,tag::scalar,tag::scalar>
: boost::mpl::false_
{
};

template <typename T>
struct is_blas_dispatchable_helper<T,T,tag::scalar,tag::scalar>
: supports_blas<typename boost::remove_const<T>::type>::type
{
};

template <typename T1,typename T2>
struct is_blas_dispatchable
: is_blas_dispatchable_helper<T1,T2, typename get_entity<T1>::type, typename get_entity<T2>::type>::type
{
};


} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_IS_BLAS_DISPATCHABLE_HPP
