/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>
#include <alps/alea/computed.hpp>

#include <memory>

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
 * "union"-like structure, which usually represents the sum (in the case of
 * accumulating) or the mean (in the case of working with the mean).
 */
template <typename T>
class mean_data
{
public:
    /** Constructs new data with size elements */
    mean_data(size_t size) : data_(size) { reset(); }

    /** Resets all data to zeros */
    void reset();

    /** Returns the size of the result vector */
    size_t size() const { return data_.rows(); }

    /** Returns number of accumulated data points */
    const size_t &count() const { return count_; }

    /** Returns number of accumulated data points */
    size_t &count() { return count_; }

    /** Returns data vector (either mean or sum) */
    const column<T> &data() const { return data_; }

    /** Returns data vector (either mean or sum) */
    column<T> &data() { return data_; }

    /** Re-interprets data that was a sum as mean */
    void convert_to_mean();

    /** Re-interprets data that was a mean as sum */
    void convert_to_sum();

private:
    column<T> data_;
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
    mean_acc() : store_(), size_(-1) { }

    mean_acc(size_t size) : store_(new mean_data<T>(size)), size_(size) { }

    mean_acc(const mean_acc &other);

    mean_acc &operator=(const mean_acc &other);

    void reset();

    bool initialized() const { return size_ != (size_t)-1; }

    bool valid() const { return (bool)store_; }

    size_t size() const { return size_; }

    template <typename S>
    mean_acc &operator<<(const S &obj)
    {
        computed_adapter<T, S> source(obj);
        return *this << (const computed<T> &) source;
    }

    mean_acc &operator<<(const computed<T> &source);

    size_t count() const { return store_->count(); }

    mean_result<T> result() const;

    mean_result<T> finalize();

    const mean_data<T> &store() const { return *store_; }

protected:
    void finalize_to(mean_result<T> &result);

private:
    std::unique_ptr< mean_data<T> > store_;
    size_t size_;
};

template <typename T>
struct traits< mean_acc<T> >
{
    typedef T value_type;
    typedef mean_result<T> result_type;
};

extern template class mean_acc<double>;
extern template class mean_acc<std::complex<double> >;


/**
 * Result of a mean accumulation
 */
template <typename T>
class mean_result
{
public:
    mean_result() { }

    mean_result(const mean_data<T> &acc_data)
        : store_(new mean_data<T>(acc_data))
    { }

    bool initialized() const { return true; }

    bool valid() const { return (bool)store_; }

    size_t size() const { return store_->size(); }

    size_t count() const { return store_->count(); }

    const column<T> &mean() const { return store_->data(); }

    const mean_data<T> &store() const { return *store_; }

    mean_data<T> &store() { return *store_; }

    void reduce(reducer &);

    void serialize(serializer &);

private:
    std::unique_ptr< mean_data<T> > store_;

    friend class mean_acc<T>;
};

template <typename T>
struct traits< mean_result<T> >
{
    typedef T value_type;
};

extern template class mean_result<double>;
extern template class mean_result<std::complex<double> >;

}}
