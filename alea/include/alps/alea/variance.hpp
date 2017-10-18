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
 * Data for variance accumulation.
 *
 * As with `mean_acc`, this class is basically a "tagged union"-like structure,
 * which for a data series `(X[0], ... X[count_-1])` either represents the sum
 * of X[i] and the sum of X[i]*X[i] (sum state) or the sample mean and sample
 * variance of X (mean state).
 *
 * This is "switch" is useful for result handling, both in intermediate steps
 * of the calculation, where it avoids temporaries; and for reduction over the
 * network.
 */
template <typename T, typename Strategy=circular_var<T> >
class var_data
{
public:
    typedef typename Strategy::value_type value_type;
    typedef typename Strategy::var_type var_type;

public:
    var_data(size_t size=0);

    void reset();

    size_t size() const { return data_.rows(); }

    const size_t &count() const { return count_; }

    size_t &count() { return count_; }

    const column<value_type> &data() const { return data_; }

    column<value_type> &data() { return data_; }

    const column<var_type> &data2() const { return data2_; }

    column<var_type> &data2() { return data2_; }

    size_t state() const { return state_; }

    void unlock_mean() const;

    void unlock_sum() const;

protected:
    void state(data_state new_state) const { state_ = new_state; }

private:
    mutable column<T> data_;
    mutable column<var_type> data2_;
    mutable data_state state_;
    size_t count_;
};

template <typename T, typename Strategy>
struct traits< var_data<T,Strategy> >
{
    typedef typename Strategy::value_type value_type;
    typedef typename Strategy::var_type var_type;
};

extern template class var_data<double>;
extern template class var_data<std::complex<double> >;
extern template class var_data<std::complex<double>, elliptic_var<std::complex<double> > >;

/**
 * Accumulator which tracks the mean and a naive variance estimate.
 */
template <typename T, typename Strategy=circular_var<T> >
class var_acc
{
public:
    typedef typename Strategy::value_type value_type;
    typedef typename Strategy::var_type var_type;
    typedef computed_cmember<value_type, var_acc> mresult;
    typedef computed_cmember<var_type, var_acc> vresult;

public:
    var_acc(size_t size, size_t bundle_size=1);

    void reset();

    size_t size() const { return store_.size(); }

    template <typename S>
    var_acc &operator<<(const S &obj)
    {
        computed_adapter<value_type, S> source(obj);
        return *this << (const computed<value_type> &) source;
    }

    var_acc &operator<<(const computed<value_type> &source);

    size_t count() const { return store_.count(); }

    const column<value_type> &mean() const { store_.unlock_mean(); return store_.data(); }

    const column<var_type> &var() const { store_.unlock_mean(); return store_.data2(); }

    vresult stderr() const { return vresult(*this, &var_acc::get_stderr, size()); }

    void uplevel(var_acc &new_uplevel) { uplevel_ = &new_uplevel; }

    const bundle<value_type> &current() const { return current_; }

    const var_data<value_type, Strategy> &store() const { return store_; }

protected:
    void get_stderr(sink<var_type> out) const;

    void add_bundle();

private:
    bundle<value_type> current_;
    var_data<value_type, Strategy> store_;
    var_acc *uplevel_;
};

template <typename T, typename Strategy>
struct traits< var_acc<T,Strategy> >
{
    typedef typename Strategy::value_type value_type;
    typedef typename Strategy::var_type var_type;
};

extern template class var_acc<double>;
extern template class var_acc<std::complex<double> >;
extern template class var_acc<std::complex<double>, elliptic_var<std::complex<double> > >;

}} /* namespace alps::alea */
