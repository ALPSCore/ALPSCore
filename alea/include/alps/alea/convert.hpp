/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/joined.hpp>

namespace alps { namespace alea {

/**
 * Joins two statistical results together, assuming their mutual independence.
 *
 * Returns a result of the random vector which is the concatenation of the
 * random vectors corresponding to the arguments `first`, `second`, i.e., of
 * size ` first.size() + second.size()`.   Assumes that `first` and `second`
 * are uncorrelated (their covariance is zero).
 *
 * One can combine results of different type; in this case, `Result` is
 * inferred in such a way that as much information as possible is preserved
 * from the constituent accumulators.
 */
template <typename R1, typename R2,
          typename Result=typename internal::joined<R1, R2>::result_type>
Result join(const R1 &first, const R2 &second);

template <typename R1, typename R2, typename T>
mean_result<T> join(const R1 &first, const R2 &second)
{
    mean_data<T> res(first.size() + second.size());
    res.store().data().topRows(first.size()) = first.store().data();
    res.store().data().bottomRows(second.size()) = second.store().data();
    res.store().count() = 1;   // TODO: does this make sense?
    return res;
}

template <typename R1, typename R2, typename T, typename Str>
var_result<T,Str> join(const R1 &first, const R2 &second)
{
    throw std::runtime_error("Not implemented");   // FIXME
}

template <typename R1, typename R2, typename T, typename Str>
cov_result<T,Str> join(const R1 &first, const R2 &second)
{
    throw std::runtime_error("Not implemented");   // FIXME
}

template <typename R1, typename R2, typename T, typename Str>
autocorr_result<T> join(const R1 &first, const R2 &second)
{
    throw std::runtime_error("Not implemented");   // FIXME
}

template <typename R1, typename R2, typename T, typename Str>
batch_result<T> join(const R1 &first, const R2 &second)
{
    throw std::runtime_error("Not implemented");   // FIXME
}

}}
