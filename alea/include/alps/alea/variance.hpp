/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/bundle.hpp>
#include <alps/alea/complex_op.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/var_strategy.hpp>

#include <memory>

// Forward declarations

namespace alps { namespace alea {
    template <typename T, typename Str> class var_data;
    template <typename T, typename Str> class var_acc;
    template <typename T, typename Str> class var_result;

    template <typename T> class autocorr_acc;
    template <typename T> class autocorr_result;

    template <typename T> class batch_result;

    template <typename T, typename Str>
    void serialize(serializer &, const std::string &, const var_result<T,Str> &);

    template <typename T, typename Str>
    void deserialize(deserializer &, const std::string &, var_result<T,Str> &);

    template <typename T, typename Str>
    std::ostream &operator<<(std::ostream &, const var_result<T,Str> &);
}}

// Actual declarations

namespace alps { namespace alea {

/**
 * Data for variance accumulation.
 *
 * As with `mean_acc`, this class is basically a "union"-like structure,
 * which for a data series `(X[0], ... X[count_-1])` either represents the sum
 * of X[i] and the sum of X[i]*X[i] (sum state) or the sample mean and sample
 * variance of X (mean state).
 */
template <typename T, typename Strategy=circular_var>
class var_data
{
public:
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;

public:
    var_data(size_t size);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return data_.rows(); }

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return count_; }

    /** Returns sample size, i.e., number of accumulated data points */
    size_t &count() { return count_; }

    /** Returns sum of squared weights */
    double count2() const { return count2_; }

    /** Returns sum of squared weights */
    double &count2() { return count2_; }

    const column<value_type> &data() const { return data_; }

    column<value_type> &data() { return data_; }

    const column<var_type> &data2() const { return data2_; }

    column<var_type> &data2() { return data2_; }

    void convert_to_mean();

    void convert_to_sum();

private:
    column<T> data_;
    column<var_type> data2_;
    size_t count_;
    double count2_;

    friend class var_acc<T, Strategy>;
    friend class var_result<T, Strategy>;
    friend void serialize<>(serializer &, const std::string &, const var_result<T,Strategy> &);
    friend void deserialize<>(deserializer &, const std::string &, var_result<T,Strategy> &);
};

template <typename T, typename Strategy>
struct traits< var_data<T,Strategy> >
{
    typedef Strategy strategy_type;
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
};

extern template class var_data<double>;
extern template class var_data<std::complex<double>, circular_var>;
extern template class var_data<std::complex<double>, elliptic_var>;

/**
 * Accumulator which tracks the mean and a naive variance estimate.
 */
template <typename T, typename Strategy=circular_var>
class var_acc
{
public:
    using value_type = T;
    using var_type = typename bind<Strategy, T>::var_type;

public:
    var_acc(size_t size=1, size_t batch_size=1);

    var_acc(const var_acc &other);

    var_acc &operator=(const var_acc &other);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    /** Update the size and discard all measurements, if any */
    void set_size(size_t size);

    /** Update the batch size and discard current batch */
    void set_batch_size(size_t batch_size);

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return current_.size(); }

    /** Returns number of data points per batch */
    size_t batch_size() const { return current_.target(); }

    /** Add computed vector to the accumulator */
    var_acc &operator<<(const computed<T> &src) { add(src, 1, nullptr); return *this; }

    /** Merge partial result into accumulator */
    var_acc &operator<<(const var_result<T,Strategy> &result);

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return store_->count(); }

    /** Returns result corresponding to current state of accumulator */
    var_result<T,Strategy> result() const;

    /** Frees data associated with accumulator and return result */
    var_result<T,Strategy> finalize();

    const bundle<value_type> &current() const { return current_; }

    /** Return backend object used for storing estimands */
    const var_data<T,Strategy> &store() const { return *store_; }

protected:
    void add(const computed<T> &source, size_t count, var_acc *cascade);

    void add_bundle(var_acc *cascade);

    void finalize_to(var_result<T,Strategy> &result, var_acc *cascade);

private:
    std::unique_ptr< var_data<value_type, Strategy> > store_;
    bundle<value_type> current_;

    friend class autocorr_acc<T>;
    friend class batch_result<T>;
};

template <typename T, typename Strategy>
struct traits< var_acc<T,Strategy> >
{
    typedef Strategy strategy_type;
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
    typedef var_result<T,Strategy> result_type;
    typedef var_data<T, Strategy> store_type;
};

extern template class var_acc<double>;
extern template class var_acc<std::complex<double>, circular_var>;
extern template class var_acc<std::complex<double>, elliptic_var>;

/**
 * Result which contains mean and a naive variance estimate.
 */
template <typename T, typename Strategy=circular_var>
class var_result
{
public:
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;

public:
    var_result() { }

    var_result(const var_data<T,Strategy> &acc_data)
        : store_(new var_data<T,Strategy>(acc_data))
    { }

    var_result(const var_result &other);

    var_result &operator=(const var_result &other);

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return store_->size(); }

    /** Returns average batch size */
    double batch_size() const { return store_->count2() / store_->count(); }

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return store_->count(); }

    /** Returns sum of squared sample sizes */
    double count2() const { return store_->count2(); }

    /** Returns effective number of observations */
    double observations() const { return count() / batch_size(); }

    /** Returns sample mean */
    const column<T> &mean() const { return store_->data(); }

    // TODO: this is essentially a weighted variance thing.  The weighted
    // variance differs from the pooled on by a factor.  We should probably
    // split the two things.

    /** Returns bias-corrected sample variance */
    column<var_type> var() const { return batch_size() * store_->data2(); }

    /** Returns bias-corrected standard error of the mean */
    column<var_type> stderror() const;

    /** Return backend object used for storing estimands */
    const var_data<T,Strategy> &store() const { return *store_; }

    /** Return backend object used for storing estimands */
    var_data<T,Strategy> &store() { return *store_; }

    /** Collect measurements from different instances using sum-reducer */
    void reduce(const reducer &r) { reduce(r, true, true); }

    /** Convert result to a permanent format (write to disk etc.) */
    friend void serialize<>(serializer &, const std::string &, const var_result &);

    /** Convert result from a permanent format (write to disk etc.) */
    friend void deserialize<>(deserializer &, const std::string &, var_result &);

    /** Write some info about the result to a stream */
    friend std::ostream &operator<< <>(std::ostream &, const var_result &);

protected:
    void reduce(const reducer &, bool do_pre_commit, bool do_post_commit);

private:
    std::unique_ptr< var_data<T,Strategy> > store_;

    friend class var_acc<T,Strategy>;
    friend class autocorr_result<T>;
};

/** Check if two results are identical */
template <typename T, typename Strategy>
bool operator==(const var_result<T, Strategy> &r1, const var_result<T, Strategy> &r2);
template <typename T, typename Strategy>
bool operator!=(const var_result<T, Strategy> &r1, const var_result<T, Strategy> &r2)
{
    return !operator==(r1, r2);
}

template<typename T> struct is_alea_acc<var_acc<T, circular_var>> :
    std::true_type {};
template<typename T> struct is_alea_acc<var_acc<T, elliptic_var>> :
    std::true_type {};
template<typename T> struct is_alea_result<var_result<T, circular_var>> :
    std::true_type {};
template<typename T> struct is_alea_result<var_result<T, elliptic_var>> :
    std::true_type {};

template <typename T, typename Strategy>
struct traits< var_result<T,Strategy> >
{
    typedef Strategy strategy_type;
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;

    const static bool HAVE_MEAN  = true;
    const static bool HAVE_VAR   = true;
    const static bool HAVE_COV   = false;
    const static bool HAVE_TAU   = false;
    const static bool HAVE_BATCH = false;
};

extern template class var_result<double>;
extern template class var_result<std::complex<double>, circular_var>;
extern template class var_result<std::complex<double>, elliptic_var>;

}} /* namespace alps::alea */
