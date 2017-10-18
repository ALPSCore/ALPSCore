/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/computed.hpp>
#include <alps/alea/util.hpp>

// Forward declarations

namespace alps { namespace alea {
    template <typename T> class batch_acc;
    class galois_hopper;
}}

namespace alps { namespace alea {

/**
 * Continuous bin merging strategy that preserves time ordering.
 *
 * We want to split a time series (t1, ..., tN) into n compact batches.  If we
 * don't know the number of measurements beforehand, the naive strategy to
 * proceed is to average over n/2 pair of batches whenever we run out of space.
 * However, this loses half the batch information.
 *
 * This class solves the problem by proposing a merge of one batch into it
 * successor at every step, freeing exactly one spot and thus preserving the
 * number of bins.  One example usage is:
 *
 *     galois_hopper x(size);
 *     while (true) {
 *         if (x.merge_mode()) {
 *             batch[x.merge_into()] += batch[x.current()];
 *             batch[x.current()] = 0;
 *         }
 *         for (size_t i = 0; i != x.factor(); ++i) {
 *             value = get_next_value();
 *             batch[x.current()] += value;
 *         }
 *         x.advance();
 *     }
 */
class galois_hopper
{
public:
    /** Expects number of batches */
    galois_hopper(size_t size);

    /** Advance to the next prescription */
    galois_hopper &operator++();

    /** Advance to the next prescription */
    galois_hopper operator++(int);

    /** Reset */
    void reset(bool merge_mode=false);

    /** Current batch to fill */
    size_t current() const { return current_; }

    /** Are we in merge mode? */
    bool merge_mode() const { return level_ != 0; }

    /** Merge current batch into this one before filling */
    size_t merge_into() const { return (current_ + skip_) % (size_ + 1); }

    /** Scaling factor of bin size (2**level) */
    double factor() const { return factor_; }

    /** Merging level */
    size_t level() const { return level_; }

    /** Galois cycle */
    size_t cycle() const { return cycle_; }

private:
    void advance_fill();
    void advance_galois();

    size_t size_;
    size_t level_, factor_;
    size_t current_, skip_, level_pos_, cycle_;
};

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

    const galois_hopper &cursor() const { return cursor_; }

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
    galois_hopper cursor_;
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
