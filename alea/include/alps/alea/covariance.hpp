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

namespace alps { namespace alea {

/**
 * Data for covariance accumulation.
 *
 * As with `mean_acc`, this class is basically a "tagged union"-like structure,
 * which for a data series `(X[0], ... X[count_-1])` either represents the sum
 * of X[i] and the sum of X[i]*X[j] (sum state) or the sample mean and sample
 * covariance of X (mean state).
 *
 * This is "switch" is useful for result handling, both in intermediate steps
 * of the calculation, where it avoids temporaries; and for reduction over the
 * network.
 */
template <typename T, typename Strategy=circular_var<T> >
class cov_data
{
public:
    typedef typename Strategy::value_type value_type;
    typedef typename Strategy::cov_type cov_type;
    typedef typename eigen<cov_type>::matrix cov_matrix_type;

public:
    cov_data(size_t size);

    void reset();

    size_t size() const { return data_.rows(); }

    const size_t &count() const { return count_; }

    size_t &count() { return count_; }

    const column<value_type> &data() const { return data_; }

    column<value_type> &data() { return data_; }

    const cov_matrix_type &data2() const { return data2_; }

    cov_matrix_type &data2() { return data2_; }

    size_t state() const { return state_; }

    void unlock_mean() const;

    void unlock_sum() const;

protected:
    void state(data_state new_state) const { state_ = new_state; }

private:
    mutable column<T> data_;
    mutable cov_matrix_type data2_;
    mutable data_state state_;
    size_t count_;
};

extern template class cov_data<double>;
extern template class cov_data<std::complex<double> >;
extern template class cov_data<std::complex<double>, elliptic_var<std::complex<double> > >;

/**
 * Accumulator which tracks the mean and a naive covariance estimate.
 */
template <typename T, typename Strategy=circular_var<T> >
class cov_acc
{
public:
    typedef T value_type;
    typedef typename Strategy::var_type var_type;
    typedef typename Strategy::cov_type cov_type;
    typedef typename eigen<cov_type>::matrix cov_matrix_type;

    typedef computed_cmember<T, cov_acc> result;
    typedef computed_cmember<var_type, cov_acc> vresult;

public:
    cov_acc(size_t size, size_t bundle_size=1);

    size_t size() const { return store_.size(); }

    template <typename S>
    cov_acc &operator<<(const S &obj)
    {
        computed_adapter<T, S> source(obj);
        return *this << (computed<T> &) source;
    }

    cov_acc &operator<<(computed<T> &source);

    void reset();

    size_t count() const { return store_.count(); }

    const column<value_type> &mean() const { store_.unlock_mean(); return store_.data(); }

    const cov_matrix_type &cov() const { store_.unlock_mean(); return store_.data2(); }

    vresult var() const { return vresult(*this, &cov_acc::get_var, size()); }

    vresult stderr() const { return vresult(*this, &cov_acc::get_stderr, size()); }

    void uplevel(cov_acc &new_uplevel) { uplevel_ = &new_uplevel; }

    const bundle<value_type> &current() const { return current_; }

    const cov_data<value_type, Strategy> &store() const { return store_; }

protected:
    void get_var(sink<var_type> out) const;

    void get_stderr(sink<var_type> out) const;

    void add_bundle();

private:
    bundle<value_type> current_;
    cov_data<value_type, Strategy> store_;
    cov_acc *uplevel_;
};

extern template class cov_acc<double>;
extern template class cov_acc<std::complex<double> >;
extern template class cov_acc<std::complex<double>, elliptic_var<std::complex<double> > >;


}}
