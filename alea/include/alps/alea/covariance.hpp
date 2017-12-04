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

    void reset();

    size_t size() const { return data_.rows(); }

    size_t count() const { return count_; }

    size_t &count() { return count_; }

    const column<value_type> &data() const { return data_; }

    column<value_type> &data() { return data_; }

    const cov_matrix_type &data2() const { return data2_; }

    cov_matrix_type &data2() { return data2_; }

    void convert_to_mean();

    void convert_to_sum();

private:
    column<T> data_;
    cov_matrix_type data2_;
    size_t count_;
};

template <typename T, typename Strategy>
struct traits< cov_data<T,Strategy> >
{
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

    void reset();

    bool valid() const { return (bool)store_; }

    size_t size() const { return current_.size(); }

    template <typename S>
    cov_acc &operator<<(const S &obj)
    {
        computed_adapter<value_type, S> source(obj);
        return *this << (const computed<value_type> &) source;
    }

    cov_acc &operator<<(const computed<value_type> &source);

    size_t count() const { return store_->count(); }

    cov_result<T,Strategy> result() const;

    cov_result<T,Strategy> finalize();

    const bundle<value_type> &current() const { return current_; }

    const cov_data<T,Strategy> &store() const { return *store_; }

protected:
    void add_bundle();

    void uplevel(cov_acc &new_uplevel) { uplevel_ = &new_uplevel; }

    void finalize_to(cov_result<T,Strategy> &result);

private:
    std::unique_ptr<cov_data<T,Strategy> > store_;
    bundle<value_type> current_;
    cov_acc *uplevel_;
};

template <typename T, typename Strategy>
struct traits< cov_acc<T,Strategy> >
{
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

    bool valid() const { return (bool)store_; }

    size_t size() const { return store_->size(); }

    size_t count() const { return store_->count(); }

    const column<T> &mean() const { return store_->data(); }

    column<var_type> var() const { return store_->data2().diagonal().real(); }

    const typename eigen<cov_type>::matrix &cov() const { return store_->data2(); }

    column<var_type> stderror() const;

    const cov_data<T,Strategy> &store() const { return *store_; }

    cov_data<T,Strategy> &store() { return *store_; }

    void reduce(reducer &);

    void serialize(serializer &);

private:
    std::unique_ptr<cov_data<T,Strategy> > store_;

    friend class cov_acc<T,Strategy>;
};

template <typename T, typename Strategy>
struct traits< cov_result<T,Strategy> >
{
    typedef typename bind<Strategy, T>::value_type value_type;
    typedef typename bind<Strategy, T>::var_type var_type;
};

extern template class cov_result<double>;
extern template class cov_result<std::complex<double>, circular_var>;
extern template class cov_result<std::complex<double>, elliptic_var>;


}}
