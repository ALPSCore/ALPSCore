/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/computed.hpp>

// Forward declarations

namespace alps { namespace alea {
    template <typename T> class mean_data;
    template <typename T> class mean_acc;
    template <typename T> class mean_result;
}}

// Actual declarations

namespace alps { namespace alea {

/**
 * Data which tracks only the mean/sum and the count.
 *
 * Mean data may be particularly memory-constrained, therefore mean_acc is a
 * "tagged union"-like structure, which either represents the sum (in the case
 * of accumulating) or the mean (in the case of working with the mean).
 */
template <typename T>
class mean_data
{
public:
    mean_data(size_t size) : data_(size) { reset(); }

    void reset();

    size_t size() const { return data_.rows(); }

    const size_t &count() const { return count_; }

    size_t &count() { return count_; }

    const column<T> &data() const { return data_; }

    column<T> &data() { return data_; }

    data_state state() const { return state_; }

    void unlock_mean();

    void unlock_sum();

protected:
    void state(data_state new_state);

private:
    column<T> data_;
    data_state state_;
    size_t count_;
};

template <typename T>
struct traits< mean_data<T> >
{
    typedef T value_type;
};

extern template class mean_data<double>;
extern template class mean_data<std::complex<double> >;


/**
 * Accumulator which tracks only the mean.
 */
template <typename T>
class mean_acc
{
public:
    mean_acc(size_t size=0) : store_(size) { }

    template <typename S>
    mean_acc &operator<<(const S &obj)
    {
        computed_adapter<T, S> source(obj);
        return *this << (const computed<T> &) source;
    }

    mean_acc &operator<<(const computed<T> &source);

    size_t count() const { return store_.count(); }

    const column<T> &mean() const { store_.unlock_mean(); return store_.data(); }

    const column<T> &sum() const { store_.unlock_sum(); return store_.data(); }

    mean_result<T> result() const { return mean_result<T>(*this); }

protected:
    const mean_data<T> &store() const { return store_; }

    mean_data<T> &store() { return store_; }

private:
    mutable mean_data<T> store_;    // we need to switch between mean/sum

    friend class mean_result<T>;
};

template <typename T>
struct traits< mean_acc<T> >
{
    typedef T value_type;
};

extern template class mean_acc<double>;
extern template class mean_acc<std::complex<double> >;

// mean_result<T>

/**
 * Mean result
 */
template <typename T>
class mean_result
{
public:
    mean_result() : store_(0) { }

    mean_result(const mean_acc<T> &acc) : store_(acc.store()) { }

    size_t count() const { return store_.count(); }

    const column<T> &mean() const { store_.unlock_mean(); return store_.data(); }

    const column<T> &sum() const { store_.unlock_sum(); return store_.data(); }

    //void serialize(serializer &s) const;

    void reduce(reducer &r);

    const mean_data<T> &store() const { return store_; }

    mean_data<T> &store() { return store_; }

private:
    mutable mean_data<T> store_;
};

extern template class mean_result<double>;
extern template class mean_result<std::complex<double> >;

}}
