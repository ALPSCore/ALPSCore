/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

// We only need forwards here

namespace alps { namespace alea {
    template <typename T> class mean_result;
    template <typename T, typename Str> class var_result;
    template <typename T, typename Str> class cov_result;
    template <typename T> class autocorr_result;
    template <typename T> class batch_result;

    struct circular_var;
    struct elliptic_var;

}}

namespace alps { namespace alea { namespace internal {
    template <typename... Pack> struct add_scalar;
    template <typename... Pack> struct joined_value;
}}}

// Actual declarations

namespace alps { namespace alea { namespace internal {

/** Given a set of types T1, T2, ..., return type of T1() + T2() + ... */
template <typename... Pack>
using add_scalar_type = typename add_scalar<Pack...>::type;

/** Given a set of types T1, T2, ..., return type of T1() + T2() + ... */
template <typename T>
struct add_scalar<T>
{
    using type = T;
};

/** Given a set of types T1, T2, ..., return type of T1() + T2() + ... */
template <typename Head, typename... Tail>
struct add_scalar<Head, Tail...>
{
    using type = decltype(Head(0) + typename add_scalar<Tail...>::type(0));
};

/** Given a set of result types R1, R2, ..., return value type of joined result */
template <typename... Pack>
using joined_value_type = typename joined_value<Pack...>::type;

/** Given a set of result types R1, R2, ..., return value type of joined result */
template <typename T>
struct joined_value<T>
{
    using type = typename traits<T>::value_type;
};

/** Given a set of result types R1, R2, ..., return value type of joined result */
template <typename Head, typename... Tail>
struct joined_value<Head, Tail...>
{
    using type = decltype(typename joined_value<Head>::type(0)
                          + typename joined_value<Tail...>::type(0));
};

// Predicates

/** Given a set of result traits, is the joined result a batch_result */
template <typename T1, typename T2>
constexpr bool joins_batch()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
        && T1::HAVE_BATCH && T2::HAVE_BATCH;
}

/** Given a set of result traits, is the joined result a autocor_result */
template <typename T1, typename T2>
constexpr bool joins_autocorr()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
        && T1::HAVE_TAU && T2::HAVE_TAU;
}

/** Given a set of result traits, is the joined result a cov_result */
template <typename T1, typename T2>
constexpr bool joins_cov()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
           && T1::HAVE_COV && T2::HAVE_COV
           && !(T1::HAVE_BATCH && T2::HAVE_BATCH);
}

/** Given a set of result traits, is the joined result a var_result */
template <typename T1, typename T2>
constexpr bool joins_var()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
           && (T1::HAVE_VAR && T2::HAVE_VAR)
           && !(T1::HAVE_COV && T2::HAVE_COV)
           && !(T1::HAVE_TAU && T2::HAVE_TAU)
           && !(T1::HAVE_BATCH && T2::HAVE_BATCH);
}

/** Given a set of result traits, is the joined result a mean_result */
template <typename T1, typename T2>
constexpr bool joins_mean()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
           && T1::HAVE_MEAN && T2::HAVE_MEAN
           && !(T1::HAVE_VAR && T2::HAVE_VAR);
}

/**
 * Determines the "greatest common denominator" type when combining results.
 */
template <typename R1, typename R2, typename Enabler=void>
struct joined;

template <typename R1, typename R2>
struct joined<R1, R2,
        typename std::enable_if<joins_batch<traits<R1>, traits<R2> >()>::type>
{
    typedef batch_result<typename traits<R1>::value_type> result_type;
};

template <typename R1, typename R2>
struct joined<R1, R2,
        typename std::enable_if<joins_autocorr<traits<R1>, traits<R2> >()>::type>
{
    typedef autocorr_result<typename traits<R1>::value_type> result_type;
};

template <typename R1, typename R2>
struct joined<R1, R2,
        typename std::enable_if<joins_cov<traits<R1>, traits<R2> >()>::type>
{
    typedef cov_result<typename traits<R1>::value_type,
                       typename traits<R1>::strategy_type> result_type;
};

template <typename R1, typename R2>
struct joined<R1, R2,
        typename std::enable_if<joins_var<traits<R1>, traits<R2> >()>::type>
{
    typedef var_result<typename traits<R1>::value_type,
                       typename traits<R1>::strategy_type> result_type;
};

template <typename R1, typename R2>
struct joined<R1, R2,
        typename std::enable_if<joins_mean<traits<R1>, traits<R2> >()>::type>
{
    typedef mean_result<typename traits<R1>::value_type> result_type;
};

}}}   /* namespace alps::alea::internal */
