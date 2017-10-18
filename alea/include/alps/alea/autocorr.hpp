/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/variance.hpp>

#include <list>

namespace alps { namespace alea {

/**
 * Accumulator for the integrated autocorrelation time.
 *
 * The integrated autocorrelation time `tau_int` of a time series is defined
 * as the large-n limit of:
 *
 *                   1 + 2 * tau_int = n * var(n) / var(1),                (A)
 *
 * where `var(n)` is the sample variance obtained when averaging over batches
 * batches, each batch being the mean of `n` consecutive elements of the series.
 * Given a simulation of `N` steps, its corresponding squared error `sq_error`
 * must thus be corrected as:
 *
 *                sq_error = (1 + 2 * tau_int) * var(1) / N                (B)
 *
 * which can be seen as replacing `N` with the number of uncorrelated samples.
 * For a finite simulation, a tradeoff must be made between
 *
 *   (1) formal validity of above equations, which improves with `n`,
 *   (2) statistical uncertainty in `tau_int`, which improves with `N/n`.
 *
 * This can be seen by plugging (A) into (B), which just yields the normal
 * error estimate when sampling over bins of size `n`.
 *
 * The class builds up a hierarchy of variance estimates for different batch
 * sizes, starting with `n=batch_size` at level 0, and increasing by a factor
 * `granularity` at each level. Assuming `k`-sized vectors, the estimator
 * scales as `O(k * log N)` in memory and `O(k * N * log log N)` in runtime.
 */
template <typename T>
class autocorr_acc
{
public:
    typedef T value_type;
    typedef typename make_real<T>::type var_type;

    typedef computed_cmember<value_type, autocorr_acc> result;
    typedef computed_cmember<var_type, autocorr_acc> eresult;

public:
    autocorr_acc(size_t size, size_t batch_size=1, size_t granularity=2);

    size_t size() const { return level_.begin()->size(); }

    template <typename S>
    autocorr_acc &operator<<(const S &obj)
    {
        computed_adapter<T, S> source(obj);
        return *this << (computed<T> &) source;
    }

    autocorr_acc &operator<<(computed<T> &source);

    void reset();

    size_t count() const { return level_[0].count(); }

    size_t num_level() const { return level_.size(); }

    const var_acc<T> &level(size_t i) const { return level_[i]; }

    size_t find_level(size_t min_samples) const;

    size_t batch_size(size_t level) const;

    const column<value_type> &mean() const;

    const column<var_type> &var() const;

    eresult stderr() { return eresult(*this, &autocorr_acc::get_stderr, size()); }

    eresult tau() { return eresult(*this, &autocorr_acc::get_tau, size()); }

    size_t nextlevel() const { return nextlevel_; }

protected:
    void get_stderr(sink<var_type> out) const;

    void get_tau(sink<var_type> out) const;

    void add_level();

private:
    size_t count_, nextlevel_, granularity_;
    std::vector< var_acc<T> > level_;    // dangerous, but convenient ...
};


template <typename T>
struct traits< autocorr_acc<T> >
{
    typedef T value_type;
    typedef typename make_real<T>::type var_type;
    typedef T cov_type;
};

extern template class autocorr_acc<double>;
extern template class autocorr_acc<std::complex<double> >;

}}
