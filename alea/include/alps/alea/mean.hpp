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

    void convert_to_mean();

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

    bool valid() const { return (bool)store_; }

    size_t size() const { return size_; }

    template <typename S>
    mean_acc &operator<<(const S &obj)
    {
        computed_adapter<T, S> source(obj);
        return *this << (const computed<T> &) source;
    }

    mean_acc &operator<<(const computed<T> &source);

    mean_result<T> result() const { return mean_result<T>(*store_); }

    mean_result<T> finalize() { return mean_result<T>(store_); }

    size_t count() const { return store_->count(); }

    const column<T> &sum() const { return store_->data(); }

    column<T> mean() const { return result().mean(); }

    const mean_data<T> &store() const { return *store_; }

private:
    std::unique_ptr< mean_data<T> > store_;
    size_t size_;
};

template <typename T>
struct traits< mean_acc<T> >
{
    typedef T value_type;
};


extern template class mean_acc<double>;
extern template class mean_acc<std::complex<double> >;


/**
 * Mean result
 */
template <typename T>
class mean_result
{
public:
    mean_result() { }

    mean_result(const mean_data<T> &d)
        : store_(new mean_data<T>(d))
    {
        store_->convert_to_mean();
    }

    mean_result(std::unique_ptr< mean_data<T> > &d)
        : store_(std::move(d))
    {
        store_->convert_to_mean();
    }

    bool valid() const { return (bool)store_; }

    size_t size() const { return store_->size(); }

    size_t count() const { return store_->count(); }

    const column<T> &mean() const { return store_->data(); }

    const mean_data<T> &store() const { return *store_; }

    void reduce(reducer &);

    void serialize(serializer &);

private:
    std::unique_ptr< mean_data<T> > store_;
};

extern template class mean_result<double>;
extern template class mean_result<std::complex<double> >;

}}
