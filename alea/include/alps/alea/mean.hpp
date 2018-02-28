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

    template <typename T>
    void serialize(serializer &, const std::string &, const mean_result<T> &);

    template <typename T>
    void deserialize(deserializer &, const std::string &, mean_result<T> &);

    template <typename T>
    std::ostream &operator<<(std::ostream &, const mean_result<T> &);
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
    size_t count() const { return count_; }

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

    friend class mean_acc<T>;
    friend class mean_result<T>;
    friend void serialize<>(serializer &, const std::string &, const mean_result<T> &);
    friend void deserialize<>(deserializer &, const std::string &, mean_result<T> &);
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
    using value_type = T;

public:
    mean_acc(size_t size=1) : store_(new mean_data<T>(size)), size_(size) { }

    mean_acc(const mean_acc &other);

    mean_acc &operator=(const mean_acc &other);

    /** Re-allocate and thus clear all accumulated data */
    void reset();

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return size_; }

    /** Add computed vector to the accumulator */
    mean_acc &operator<<(const computed<T> &src) { add(src, 1); return *this; }

    /** Merge partial result into accumulator */
    mean_acc &operator<<(const mean_result<T> &result);

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return store_->count(); }

    /** Returns result corresponding to current state of accumulator */
    mean_result<T> result() const;

    /** Frees data associated with accumulator and return result */
    mean_result<T> finalize();

    /** Return backend object used for storing estimands */
    const mean_data<T> &store() const { return *store_; }

protected:
    void add(const computed<T> &source, size_t count);

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
    typedef mean_data<T> store_type;
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

    mean_result(const mean_result &other);

    mean_result &operator=(const mean_result &other);

    /** Returns `false` if `finalize()` has been called, `true` otherwise */
    bool valid() const { return (bool)store_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return store_->size(); }

    /** Returns sample size, i.e., number of accumulated data points */
    size_t count() const { return store_->count(); }

    /** Returns sample mean */
    const column<T> &mean() const { return store_->data(); }

    /** Return backend object used for storing estimands */
    const mean_data<T> &store() const { return *store_; }

    /** Return backend object used for storing estimands */
    mean_data<T> &store() { return *store_; }

    /** Collect measurements from different instances using sum-reducer */
    void reduce(const reducer &r) { return reduce(r, true, true); }

    /** Convert result to a permanent format (write to disk etc.) */
    friend void serialize<>(serializer &, const std::string &, const mean_result &);

    /** Result to a permanent format (write to disk etc.) */
    friend void deserialize<>(deserializer &, const std::string &, mean_result &);

    /** Write some info about the result to a stream */
    friend std::ostream &operator<< <>(std::ostream &, const mean_result &);

protected:
    void reduce(const reducer &, bool do_pre_commit, bool do_post_commit);

private:
    std::unique_ptr< mean_data<T> > store_;

    friend class mean_acc<T>;
};

/** Check if two results are identical */
template <typename T>
bool operator==(const mean_result<T> &r1, const mean_result<T> &r2);
template <typename T>
bool operator!=(const mean_result<T> &r1, const mean_result<T> &r2)
{
    return !operator==(r1, r2);
}

template<typename T> struct is_alea_acc<mean_acc<T>> : std::true_type {};
template<typename T> struct is_alea_result<mean_result<T>> : std::true_type {};

template <typename T>
struct traits< mean_result<T> >
{
    typedef T value_type;

    const static bool HAVE_MEAN  = true;
    const static bool HAVE_VAR   = false;
    const static bool HAVE_COV   = false;
    const static bool HAVE_TAU   = false;
    const static bool HAVE_BATCH = false;
};

extern template class mean_result<double>;
extern template class mean_result<std::complex<double> >;

}}
