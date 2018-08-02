/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/**
 * Set of auxiliary processing functions useful for implementations
 */
#pragma once

#include <alps/alea/core.hpp>

namespace alps { namespace alea { namespace internal {

template <typename Acc>
inline void check_valid(const Acc &acc)
{
    if (!acc.valid())
        throw alps::alea::finalized_accumulator();
}


template <typename T, typename... Args>
T call_vargs(std::function<T(Args...)> func, const T *args);

template <typename T, typename... Args>
T call_vargs(std::function<T(T, Args...)> func, const T *args)
{
    // use currying to transform multi-argument function to hierarchy
    const T head = *args;
    std::function<T(Args...)> closure =
                [=](Args... tail) -> T { return func(head, tail...); };
    return call_vargs(closure, ++args);
}

template <typename T>
T call_vargs(std::function<T()> func, const T *)
{
    // unravel the currying hierarchy
    return func();
}

template <typename Acc>
typename traits<Acc>::result_type finalize(Acc &acc)
{
    typename traits<Acc>::result_type result;
    acc.finalize_to(result);
    return result;
}

template <typename Acc>
typename traits<Acc>::result_type result(const Acc &acc)
{
    typename traits<Acc>::result_type result;
    Acc copy = acc;
    copy.finalize_to(result);
    return result;
}

}}} /* namespace alps::alea::internal */
