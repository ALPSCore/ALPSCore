/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/variance.hpp>

#include <vector>

// Forward declarations

namespace alps { namespace alea {
    template <typename T> class autocorr_acc;
    template <typename T> class autocorr_result;
}}

// Actual declarations

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
    typedef typename bind<circular_var, T>::var_type var_type;
    typedef var_acc<T, circular_var> level_acc_type;

public:
    autocorr_acc(size_t size=1, size_t batch_size=1, size_t granularity=2);

    void reset();

    bool valid() const { return !level_.empty(); }

    size_t size() const { return size_; }

    template <typename S>
    autocorr_acc &operator<<(const S &obj)
    {
        computed_adapter<value_type, S> source(obj);
        return *this << (const computed<value_type> &) source;
    }

    autocorr_acc &operator<<(const computed<value_type> &source);

    size_t count() const { return count_; }

    autocorr_result<T> result() const;

    autocorr_result<T> finalize();

    size_t nlevel() const { return level_.size(); }

    const level_acc_type &level(size_t i) const { return level_[i]; }

protected:
    void add_level();

    void finalize_to(autocorr_result<T> &result);

private:
    size_t size_, batch_size_, count_, nextlevel_, granularity_;
    std::vector<level_acc_type> level_;
};

template <typename T>
struct traits< autocorr_acc<T> >
{
    typedef T value_type;
    typedef typename bind<circular_var, T>::var_type var_type;
    typedef typename bind<circular_var, T>::cov_type cov_type;
    typedef autocorr_result<T> result_type;
};

extern template class autocorr_acc<double>;
extern template class autocorr_acc<std::complex<double> >;


/**
 * Result for the integrated autocorrelation time.
 *
 * @see alps::alea::autocorr_acc
 */
template <typename T>
class autocorr_result
{
public:
    typedef T value_type;
    typedef typename bind<circular_var, T>::var_type var_type;
    typedef var_result<T, circular_var> level_result_type;

public:
    autocorr_result() { }

    bool valid() const { return !level_.empty(); }

    size_t size() const { return level_[0].size(); }

    size_t count() const { return level_[0].count(); }

    const column<T> &mean() const { return level_[0].mean(); }

    column<var_type> var() const;

    column<var_type> stderror() const;

    column<var_type> tau() const;

    void reduce(reducer &);

    void serialize(serializer &);

    size_t find_level(size_t min_samples) const;

    size_t batch_size(size_t level) const;

    size_t nlevel() const { return level_.size(); }

    const level_result_type &level(size_t i) const { return level_[i]; }

private:
    const static size_t default_min_samples = 256;
    std::vector<level_result_type> level_;

    friend class autocorr_acc<T>;
};

template <typename T>
struct traits< autocorr_result<T> >
{
    typedef T value_type;
    typedef typename bind<circular_var, T>::var_type var_type;
    typedef typename bind<circular_var, T>::cov_type cov_type;
};

extern template class autocorr_result<double>;
extern template class autocorr_result<std::complex<double> >;

}}
