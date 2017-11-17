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

namespace alps { namespace alea {

/**
 * Representation of a time series in (compact) batches.
 */
template <typename T>
class batch_data
{
public:
    batch_data(size_t size, size_t num_batches=256);

    void reset();

    size_t num_batches() const { return batch_.cols(); }

    size_t size() const { return batch_.rows(); }

    typename eigen<T>::matrix &batch() { return batch_; }

    const typename eigen<T>::matrix &batch() const { return batch_; }

    typename eigen<size_t>::row &count() { return count_; }

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
    typedef typename make_real<T>::type error_type;
    typedef computed_cmember<T, batch_acc> result;
    typedef computed_cmember<error_type, batch_acc> eresult;

public:
    batch_acc(size_t size=0, size_t num_batches=256, size_t base_size=1);

    void reset();

    size_t size() const { return data_.size(); }

    template <typename S>
    batch_acc &operator<<(const S &obj)
    {
        computed_adapter<T, S> source(obj);
        return *this << (const computed<T> &) source;
    }

    batch_acc &operator<<(const computed<T> &source);

    const internal::galois_hopper &cursor() const { return cursor_; }

    const batch_data<T> &data() const { return data_; }

    const typename eigen<size_t>::row &offset() const { return offset_; }

    size_t current_batch_size() const { return base_size_ * cursor_.factor(); }

    size_t count() const { return data_.count().sum(); }

    result mean() const { return result(*this, &batch_acc::get_mean, size()); }

    eresult var() const { return eresult(*this, &batch_acc::get_var, size()); }

protected:
    void get_mean(sink<T> out) const;

    void get_var(sink<error_type> out) const;

protected:
    void next_batch();

private:
    size_t base_size_;
    internal::galois_hopper cursor_;
    batch_data<T> data_;
    typename eigen<size_t>::row offset_;
};

template <typename T>
struct traits< batch_acc<T> >
{
    typedef T value_type;
};

extern template class batch_acc<double>;
//extern template class batch_acc<std::complex<double> >;

}}
