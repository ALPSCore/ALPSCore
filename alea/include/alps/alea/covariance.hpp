/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
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
    template <typename T, typename Str> class cov_data;
    template <typename T, typename Str> class cov_acc;
    template <typename T, typename Str> class cov_result;

    template <typename T> class batch_result;
}}

// Actual declarations

namespace alps { namespace alea {

/**
 * Data for covariance accumulation.
 *
 * As with `mean_acc`, this class is basically a "union"-like structure,
 * which for a data series `(X[0], ... X[count_-1])` either represents the sum
 * of X[i] and the sum of X[i]*X[j] (sum state) or the sample mean and sample
 * covariance of X (mean state).
 */
template <typename T, typename Strategy=circular_var>
class cov_data
{
public:
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::cov_type cov_type;
    typedef typename eigen<cov_type>::matrix cov_matrix_type;

public:
    cov_data(size_t size);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return data_.rows(); }

    /** Returns sample size, i.e., number of accumulated data points */
    double count() const { return count_; }

    /** Returns sample size, i.e., number of accumulated data points */
    double &count() { return count_; }

    /** Returns sum of squared weights */
    double count2() const { return count2_; }

    /** Returns sum of squared weights */
    double &count2() { return count2_; }

    const column<value_type> &data() const { return data_; }

    column<value_type> &data() { return data_; }

    const cov_matrix_type &data2() const { return data2_; }

    cov_matrix_type &data2() { return data2_; }

    void convert_to_mean();

    void convert_to_sum();

private:
    column<T> data_;
    cov_matrix_type data2_;
    double count_, count2_;
};

template <typename T, typename Strategy>
struct traits< cov_data<T,Strategy> >
{
    typedef Strategy strategy_type;
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
    typedef typename bind<Strategy, T>::cov_type cov_type;
};

extern template class cov_data<double>;
extern template class cov_data<std::complex<double>, circular_var>;
extern template class cov_data<std::complex<double>, elliptic_var>;


/**
 * Accumulator which tracks the mean and a naive covariance estimate.
 */
template <typename T, typename Strategy=circular_var>
class cov_acc
{
public:
    typedef T value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
    typedef typename bind<Strategy, T>::cov_type cov_type;
    typedef typename eigen<cov_type>::matrix cov_matrix_type;

public:
    cov_acc(size_t size=1, size_t bundle_size=1);

    cov_acc(const cov_acc &other);

    cov_acc &operator=(const cov_acc &other);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return current_.size(); }

    /** Returns number of data points per batch */
    size_t batch_size() const { return current_.target(); }

    /** Add computed vector to the accumulator */
    cov_acc &operator<<(const computed<T> &src) { add(src, 1); return *this; }

    /** Add Eigen vector-valued expression to accumulator */
    template <typename Derived>
    cov_acc &operator<<(const Eigen::DenseBase<Derived> &o)
    { return *this << eigen_adapter<T,Derived>(o); }

    /** Add `std::vector` to accumulator */
    cov_acc &operator<<(const std::vector<T> &o) { return *this << vector_adapter<T>(o); }

    /** Add scalar value to accumulator */
    cov_acc &operator<<(T o) { return *this << value_adapter<T>(o); }

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return store_->count(); }

    /** Returns result corresponding to current state of accumulator */
    cov_result<T,Strategy> result() const;

    /** Frees data associated with accumulator and return result */
    cov_result<T,Strategy> finalize();

    const bundle<value_type> &current() const { return current_; }

    /** Return backend object used for storing estimands */
    const cov_data<T,Strategy> &store() const { return *store_; }

protected:
    void add(const computed<T> &source, size_t count);

    void add_bundle();

    void finalize_to(cov_result<T,Strategy> &result);

private:
    std::unique_ptr<cov_data<T,Strategy> > store_;
    bundle<value_type> current_;

    friend class batch_result<T>;
};

template <typename T, typename Strategy>
struct traits< cov_acc<T,Strategy> >
{
    typedef Strategy strategy_type;
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
    typedef typename bind<Strategy, T>::cov_type cov_type;
    typedef cov_result<T,Strategy> result_type;
};

extern template class cov_acc<double>;
extern template class cov_acc<std::complex<double>, circular_var>;
extern template class cov_acc<std::complex<double>, elliptic_var>;


/**
 * Accumulator which tracks the mean and a naive variance estimate.
 */
template <typename T, typename Strategy=circular_var>
class cov_result
{
public:
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
    typedef typename bind<Strategy, T>::cov_type cov_type;

public:
    cov_result() { }

    cov_result(const cov_data<T,Strategy> &acc_data)
        : store_(new cov_data<T,Strategy>(acc_data))
    { }

    cov_result(const cov_result &other);

    cov_result &operator=(const cov_result &other);

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return store_->size(); }

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return store_->count(); }

    /** Returns average batch size */
    double batch_size() const { return store_->count2() / store_->count(); }

    /** Returns effective number of observations */
    double observations() const { return count() / batch_size(); }

    /** Returns sample mean */
    const column<T> &mean() const { return store_->data(); }

    /** Returns bias-corrected sample variance */
    column<var_type> var() const { return store_->data2().diagonal().real(); }

    /** Returns bias-corrected sample covariance matrix  */
    const typename eigen<cov_type>::matrix &cov() const { return store_->data2(); }

    /** Returns bias-corrected standard error of the mean */
    column<var_type> stderror() const;

    /** Return backend object used for storing estimands */
    const cov_data<T,Strategy> &store() const { return *store_; }

    /** Return backend object used for storing estimands */
    cov_data<T,Strategy> &store() { return *store_; }

    /** Collect measurements from different instances using sum-reducer */
    void reduce(const reducer &r) { reduce(r, true, true); }

    /** Convert result to a permanent format (write to disk etc.) */
    void serialize(serializer &) const;

protected:
    void reduce(const reducer &, bool do_pre_commit, bool do_post_commit);

private:
    std::unique_ptr<cov_data<T,Strategy> > store_;

    friend class cov_acc<T,Strategy>;
};

template <typename T, typename Strategy>
struct traits< cov_result<T,Strategy> >
{
    typedef Strategy strategy_type;
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;

    const static bool HAVE_MEAN  = true;
    const static bool HAVE_VAR   = true;
    const static bool HAVE_COV   = true;
    const static bool HAVE_TAU   = false;
    const static bool HAVE_BATCH = false;
};

extern template class cov_result<double>;
extern template class cov_result<std::complex<double>, circular_var>;
extern template class cov_result<std::complex<double>, elliptic_var>;


}}
