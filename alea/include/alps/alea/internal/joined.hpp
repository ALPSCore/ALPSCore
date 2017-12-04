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

/**
 * Determines the "greatest common denominator" type when combining results.
 */
template <typename R1, typename R2>
struct joined;

// batch_result

template <typename T>
struct joined<batch_result<T>, batch_result<T> >
{
    typedef batch_result<T> result_type;
};

// autocorr_result

template <typename T>
struct joined<autocorr_result<T>, autocorr_result<T> >
{
    typedef autocorr_result<T> result_type;
};

// cov_result

template <typename T>
struct joined<cov_result<T, circular_var>, batch_result<T> >
{
    typedef cov_result<T, circular_var> result_type;
};

template <typename T>
struct joined<batch_result<T>, cov_result<T, circular_var> >
{
    typedef cov_result<T, circular_var> result_type;
};

template <typename T, typename Str>
struct joined<cov_result<T, Str>, cov_result<T, Str> >
{
    typedef cov_result<T, Str> result_type;
};

// var_result

template <typename T, typename circular_var>
struct joined<var_result<T, circular_var>, autocorr_result<T> >
{
    typedef var_result<T, circular_var> result_type;
};

template <typename T, typename circular_var>
struct joined<autocorr_result<T>, var_result<T, circular_var>  >
{
    typedef var_result<T, circular_var> result_type;
};

template <typename T, typename Str>
struct joined<var_result<T, Str>, cov_result<T, Str> >
{
    typedef var_result<T, Str> result_type;
};

template <typename T, typename Str>
struct joined<cov_result<T, Str>, var_result<T, Str>  >
{
    typedef var_result<T, Str> result_type;
};

template <typename T, typename Str>
struct joined<var_result<T, Str>, var_result<T, Str> >
{
    typedef var_result<T, Str> result_type;
};

template <typename T, typename Str, typename R>
struct joined<var_result<T, Str>, R>
{
    // Propagate through cov
    typedef typename joined<cov_result<T, Str>, R>::result_type result_type;
};

template <typename T, typename Str, typename R>
struct joined<R, var_result<T, Str> >
{
    // Propagate through cov
    typedef typename joined<cov_result<T, Str>, R>::result_type result_type;
};

// mean_result

template <typename T, typename Str>
struct joined<mean_result<T>, var_result<T, Str> >
{
    typedef mean_result<T> result_type;
};

template <typename T, typename Str>
struct joined<var_result<T, Str>, mean_result<T> >
{
    typedef mean_result<T> result_type;
};

template <typename T>
struct joined<mean_result<T>, mean_result<T> >
{
    typedef mean_result<T> result_type;
};

template <typename T, typename R>
struct joined<mean_result<T>, R>
{
    // Propagate through var
    typedef typename joined<var_result<T, circular_var>, R>::result_type result_type;
};

template <typename T, typename R>
struct joined<R, mean_result<T> >
{
    // Flip arguments
    typedef typename joined<var_result<T, circular_var>, R>::result_type result_type;
};

}}}   /* namespace alps::alea::internal */
