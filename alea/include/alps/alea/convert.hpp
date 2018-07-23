/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

#include <alps/alea/internal/util.hpp>
#include <alps/alea/internal/joined.hpp>

// Forward

namespace alps { namespace alea {
    template <typename Result> struct joiner;

    template <typename T> class mean_dat;
    template <typename T, typename Str> class var_data;
    template <typename T, typename Str> class cov_data;
    template <typename T> class batch_data;
}}

// Actual

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
 *
 * @see alps::alea::internal::joined
 */
template <typename R1, typename R2,
          typename Result=typename internal::joined<R1, R2>::result_type>
Result join(const R1 &first, const R2 &second)
{
    return joiner<Result>()(first, second);
}

/** Helper class for joining results together */
template <typename Result>
struct joiner;

template <typename T>
struct joiner<mean_result<T> >
{
    template <typename R1, typename R2>
    mean_result<T> operator()(const R1 &first, const R2 &second)
    {
        if (first.store().count() != second.store().count())
            throw weight_mismatch();  // TODO

        mean_result<T> res(mean_data<T>(first.size() + second.size()));
        res.store().data().topRows(first.size()) = first.store().data();
        res.store().data().bottomRows(second.size()) = second.store().data();
        res.store().count() = first.store().count();
        return res;
    }
};

template <typename T>
struct joiner<var_result<T> >
{
    template <typename R1, typename R2>
    var_result<T> operator()(const R1 &first, const R2 &second)
    {
        if (first.store().count() != second.store().count())
            throw weight_mismatch();
        if (first.store().count2() != second.store().count2())
            throw weight_mismatch();

        var_result<T> res(var_data<T>(first.size() + second.size()));
        res.store().data().topRows(first.size()) = first.store().data();
        res.store().data().bottomRows(second.size()) = second.store().data();
        res.store().data2().topRows(first.size()) = first.store().data2();
        res.store().data2().bottomRows(second.size()) = second.store().data2();
        res.store().count() = first.store().count();
        res.store().count2() = first.store().count2();
        return res;
    }
};

template <typename T>
struct joiner<cov_result<T> >
{
    template <typename R1, typename R2>
    cov_result<T> operator()(const R1 &first, const R2 &second)
    {
        if (first.store().count() != second.store().count())
            throw weight_mismatch();
        if (first.store().count2() != second.store().count2())
            throw weight_mismatch();

        cov_result<T> res(cov_data<T>(first.size() + second.size()));
        res.store().data().topRows(first.size()) = first.store().data();
        res.store().data().bottomRows(second.size()) = second.store().data();

        // ignore cross correlation
        res.store().data2().topLeftCorner(first.size(), first.size())
                                                = first.store().data2();
        res.store().data2().bottomRightCorner(second.size(), second.size())
                                                = second.store().data2();
        res.store().count() = first.store().count();
        return res;
    }
};

template <typename T>
struct joiner<autocorr_result<T> >
{
    template <typename R1, typename R2>
    autocorr_result<T> operator()(const R1 &first, const R2 &second)
    {
        if (first.count() != second.count())
            throw weight_mismatch();
        if (first.nlevel() != second.nlevel())
            throw size_mismatch();

        // granularities are checked on the individual levels
        autocorr_result<T> res(first.nlevel());
        for (size_t l = 0; l != first.nlevel(); ++l)
            res.level(l) = join(first.level(l), second.level(l));
        return res;
    }
};

template <typename T>
struct joiner<batch_result<T> >
{
    template <typename R1, typename R2>
    batch_result<T> operator()(const R1 &first, const R2 &second)
    {
        if (first.store().count() != second.store().count())
            throw weight_mismatch();
        if (first.store().num_batches() != second.store().num_batches())
            throw size_mismatch();

        batch_result<T> res(batch_data<T>(first.size() + second.size(),
                                          first.store().num_batches()));

        res.store().batch().topRows(first.size()) = first.store().batch();
        res.store().batch().bottomRows(second.size()) = second.store().batch();
        res.store().count() = first.store().count();
        return res;
    }
};

}}
