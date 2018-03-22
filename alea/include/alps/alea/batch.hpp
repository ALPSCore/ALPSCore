/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/internal/galois.hpp>
#include <alps/alea/var_strategy.hpp>

#include <memory>

// Forward declarations

namespace alps { namespace alea {
    template <typename T> class batch_acc;
    template <typename T> class batch_data;
    template <typename T> class batch_result;

    template <typename T>
    void serialize(serializer &, const std::string &, const batch_result<T> &);

    template <typename T>
    void deserialize(deserializer &, const std::string &, batch_result<T> &);

    template <typename T>
    std::ostream &operator<<(std::ostream &, const batch_result<T> &);
}}

// Actual declarations

namespace alps { namespace alea {

/**
 * Representation of a time series in (compact) batches.
 */
template <typename T>
class batch_data
{
public:
    batch_data(size_t size, size_t num_batches=256);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    size_t num_batches() const { return batch_.cols(); }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return batch_.rows(); }

    typename eigen<T>::matrix &batch() { return batch_; }

    const typename eigen<T>::matrix &batch() const { return batch_; }

    /** Returns sample size (number of accumulated points) for each batch */
    typename eigen<size_t>::row &count() { return count_; }

    /** Returns sample size (number of accumulated points) for each batch */
    const typename eigen<size_t>::row &count() const { return count_; }

private:
    typename eigen<T>::matrix batch_;
    typename eigen<size_t>::row count_;
};

template <typename T>
struct traits< batch_data<T> >
{
    typedef T value_type;
};

extern template class batch_data<double>;
extern template class batch_data<std::complex<double> >;

/**
 * Accumulator which keeps track of batches of (consecutive) measurements
 */
template <typename T>
class batch_acc
{
public:
    using value_type = T;

public:
    batch_acc(size_t size=1, size_t num_batches=256, size_t base_size=1);

    batch_acc(const batch_acc &other);

    batch_acc &operator=(const batch_acc &other);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return size_; }

    /** Number of stores batches */
    size_t num_batches() const { return num_batches_; }

    /** Add computed vector to the accumulator */
    batch_acc& operator<<(const computed<T>& src){ add(src, 1); return *this; }

    /** Merge partial result into accumulator */
    batch_acc &operator<<(const batch_result<T> &result);

    /** Returns sample size, i.e., total number of accumulated data points */
    size_t count() const { return store_->count().sum(); }

    /** Returns result corresponding to current state of accumulator */
    batch_result<T> result() const;

    /** Frees data associated with accumulator and return result */
    batch_result<T> finalize();

    /** Return backend object used for storing estimands */
    const batch_data<T> &store() const { return *store_; }

    const internal::galois_hopper &cursor() const { return cursor_; }

    const typename eigen<size_t>::row &offset() const { return offset_; }

    size_t current_batch_size() const { return base_size_ * cursor_.factor(); }

protected:
    void add(const computed<T> &source, size_t count);

    void next_batch();

    void finalize_to(batch_result<T> &result);

private:
    size_t size_, num_batches_, base_size_;
    std::unique_ptr< batch_data<value_type> > store_;
    internal::galois_hopper cursor_;
    typename eigen<size_t>::row offset_;
};

template <typename T>
struct traits< batch_acc<T> >
{
    typedef T value_type;
    typedef circular_var strategy_type;
    typedef batch_result<T> result_type;
    typedef batch_data<T> store_type;
};

extern template class batch_acc<double>;
extern template class batch_acc<std::complex<double> >;


/**
 * Result which contains mean and a naive variance estimate.
 */
template <typename T>
class batch_result
{
public:
    typedef T value_type;

public:
    batch_result() { }

    batch_result(const batch_data<T> &acc_data)
        : store_(new batch_data<T>(acc_data))
    { }

    batch_result(const batch_result &other);

    batch_result &operator=(const batch_result &other);

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return store_->size(); }

    /** Number of stores batches */
    size_t num_batches() const { return store_->num_batches(); }

    /** Returns sample size, i.e., total number of accumulated data points */
    size_t count() const { return store_->count().sum(); }

    /** Returns sum of squared sample sizes */
    double count2() const { return store_->count().squaredNorm(); }

    /** Returns sample mean */
    column<T> mean() const;

    /** Returns bias-corrected sample variance for given strategy */
    template <typename Strategy=circular_var>
    column<typename bind<Strategy,T>::var_type> var() const;

    /** Returns bias-corrected sample covariance matrix for given strategy */
    template <typename Strategy=circular_var>
    typename eigen<typename bind<Strategy,T>::cov_type>::matrix cov() const;

    /** Return standard error of the mean */
    column<typename bind<circular_var,T>::var_type> stderror() const;

    /** Return backend object used for storing estimands */
    const batch_data<T> &store() const { return *store_; }

    /** Return backend object used for storing estimands */
    batch_data<T> &store() { return *store_; }

    /** Collect measurements from different instances using sum-reducer */
    void reduce(const reducer &r) { reduce(r, true, true); }

    /** Convert result to a permanent format (write to disk etc.) */
    friend void serialize<>(serializer &, const std::string &, const batch_result &);

    /** Convert result to a permanent format (write to disk etc.) */
    friend void deserialize<>(deserializer &, const std::string &, batch_result &);

    /** Write some info about the result to a stream */
    friend std::ostream &operator<< <>(std::ostream &, const batch_result &);

protected:
    void reduce(const reducer &r, bool do_pre_commit, bool do_post_commit);

private:
    std::unique_ptr< batch_data<value_type> > store_;

    friend class batch_acc<T>;
};

/** Check if two results are identical */
template <typename T>
bool operator==(const batch_result<T> &r1, const batch_result<T> &r2);
template <typename T>
bool operator!=(const batch_result<T> &r1, const batch_result<T> &r2)
{
    return !operator==(r1, r2);
}

template<typename T> struct is_alea_acc<batch_acc<T>> : std::true_type {};
template<typename T> struct is_alea_result<batch_result<T>> : std::true_type {};

template <typename T>
struct traits< batch_result<T> >
{
    typedef T value_type;
    typedef circular_var strategy_type;

    const static bool HAVE_MEAN  = true;
    const static bool HAVE_VAR   = true;
    const static bool HAVE_COV   = true;
    const static bool HAVE_TAU   = false;
    const static bool HAVE_BATCH = true;
};

extern template class batch_result<double>;
extern template class batch_result<std::complex<double> >;

}} /* namespace alps::alea */
