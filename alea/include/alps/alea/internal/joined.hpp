/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
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

// Actual declarations

namespace alps { namespace alea { namespace internal {

// Predicates

template <typename T1, typename T2>
constexpr bool joins_batch()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
        && T1::HAVE_BATCH && T2::HAVE_BATCH;
}

template <typename T1, typename T2>
constexpr bool joins_autocorr()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
        && T1::HAVE_TAU && T2::HAVE_TAU;
}

template <typename T1, typename T2>
constexpr bool joins_cov()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
           && T1::HAVE_COV && T2::HAVE_COV
           && !(T1::HAVE_BATCH && T2::HAVE_BATCH);
}

template <typename T1, typename T2>
constexpr bool joins_var()
{
    return std::is_same<typename T1::value_type, typename T2::value_type>::value
           && (T1::HAVE_VAR && T2::HAVE_VAR)
           && !(T1::HAVE_COV && T2::HAVE_COV)
           && !(T1::HAVE_TAU && T2::HAVE_TAU)
           && !(T1::HAVE_BATCH && T2::HAVE_BATCH);
}

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
