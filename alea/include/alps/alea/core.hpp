/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <complex>
#include <string>
#include <vector>
#include <initializer_list>

#include <Eigen/Dense>

#include <alps/alea/complex_op.hpp>

#include <alps/common/view.hpp>

#include <alps/serialization/core.hpp>
#include <alps/serialization/eigen.hpp>
#include <alps/serialization/util.hpp>

namespace alps { namespace alea {

using std::size_t;
using std::ptrdiff_t;
using std::uint64_t;
using std::int64_t;
using std::uint32_t;
using std::int32_t;

/** Estimator cannot add to view as the sizes are mismatched */
struct size_mismatch : public std::exception { };

/** Estimator does not support this operation */
struct unsupported_operation : public std::exception { };

/** Accumulator has lost its data */
struct finalized_accumulator : public std::exception { };

/** Accumulator has lost its data */
struct weight_mismatch : public std::exception { };

template <typename T>
struct traits;

// Metafunction to detect ALEA accumulator types
template<typename T> struct is_alea_acc : std::false_type {};

// Metafunction to detect ALEA result types
template<typename T> struct is_alea_result : std::false_type {};


// import for backwards-compatibility
template <typename T> using view = alps::common::view<T>;
template <typename T> using ndview = alps::common::ndview<T>;
using serializer = alps::serialization::serializer;
using deserializer = alps::serialization::deserializer;

namespace internal {
    using serializer_sentry = alps::serialization::serializer_sentry;
    using deserializer_sentry = alps::serialization::deserializer_sentry;
}

/**
 * Interface for a computed result (a result computed on-the-fly).
 *
 * As a trivial example, here is a vector-valued estimator of size 2 that
 * always adds the vector [1.0, -1.0] to the buffer:
 *
 *     struct trivial_computed : public computed<double>
 *     {
 *         size_t size() const { return 2; }
 *         void add_to(view<T> out) { out[0] += 1.0; out[1] -= 1.0; }
 *     }
 *
 * If a `computed` is passed to an accumulator, the accumulator will call the
 * `add_to()` method zero or more times with different buffers.  This allows
 * to avoid temporaries, as the addend can be constructed in-place, which also
 * allows for sparse data.  Also, as there is usually a bin size > 1, adding is
 * the fundamental operation.
 *
 * See also: computed_wrapper<T>
 */
template <typename T>
struct computed
{
    typedef T value_type;

    /** Number of elements of the computed result */
    virtual size_t size() const = 0;

    /** Return the shape of the data - product must equal size */
    virtual std::vector<size_t> shape() const { return std::vector<size_t>(1, size()); }

    /**
     * Add computed result data to the buffer in `out`.  If `in(i)` is the
     * `i`-th component of the estimator, do the equivalent of:
     *
     *     for (size_t i = 0; i != size(); ++i)
     *         out[i] += in(i);
     */
    virtual void add_to(view<T> out) const = 0;

    /** Returns a clone of the estimator (optional) */
    virtual computed *clone() { throw unsupported_operation(); }

    /** Destroy estimator */
    virtual ~computed() { }
};

/**
 * Shorthand for Eigen column vector
 */
template <typename T>
class column
    : public Eigen::Matrix<T, Eigen::Dynamic, 1>
{
public:
    column() : Eigen::Matrix<T, Eigen::Dynamic, 1>() {}

    column(size_t size) : Eigen::Matrix<T, Eigen::Dynamic, 1>(size) {}

    column(std::initializer_list<T> l) : column(l.size()) {
      size_t i = 0;
      for (auto x : l) {
        this->Eigen::Matrix<T,Eigen::Dynamic,1>::operator()(i) = x;
        ++i;
      }
    }

    template <typename OtherDerived>
    column(const Eigen::MatrixBase<OtherDerived>& other)
        : Eigen::Matrix<T, Eigen::Dynamic, 1>(other) { }

    template<typename OtherDerived>
    column& operator=(const Eigen::MatrixBase <OtherDerived>& other)
    {
        this->Eigen::Matrix<T, Eigen::Dynamic, 1>::operator=(other);
        return *this;
    }

    // Methods for convenience and backwards compatibility

    // TODO this prevents us from doing, which we should be at some point ...
    // template <typename T>
    // using column = Eigen::Matrix<T, Eigen::Dynamic, 1>;

    size_t size() const { return this->rows(); }

    operator std::vector<T>() const
    {
        return std::vector<T>(this->data(), this->data() + this->rows());
    }
};

/**
 * Setup information struct for the reduction
 */
struct reducer_setup
{
    /** Position of the current instance (thread no./CPU/MPI rank/etc.) */
    size_t pos;

    /** Total number of instances (thread count/MPI size/etc.) */
    uint64_t count;

    /** Reductions will yield valid result on this instance */
    bool have_result;
};

/**
 * Interface to performing sum-reduction with data from reducer source.
 *
 * Applications will typically start by checking  `get_setup()` to find out
 * whether this core/thread will get a copy of the result, but this is not
 * required
 *
 * The `reduce()` family of methods take the data view and add to it the data
 * from the reducers source (possibly by performing an MPI/OpenMP reduction or
 * gathering data from files, etc.).
 *
 * Reducers need not perform the reductions immediately (they are allowed to
 * group them for increased performance.)  A call to `commit()` marks a
 * synchronization point, after which the sum-reduced data must be available
 * on all instances which have `reducer_setup.have_result` set.
 *
 * This facade allows us to abstract away the type of reduction, but most
 * importantly does not pull in a mandatory MPI dependency for the use of
 * the accumulators.
 *
 * @see alps::alea::mpi_reducer
 */
struct reducer
{
    /** Set-up reduction operation */
    virtual reducer_setup get_setup() const = 0;

    /** Get maximum of scalar value over all instances (immediate) */
    virtual int64_t get_max(int64_t value) const = 0;

    /** Reduce double data-set into `data` */
    virtual void reduce(view<double> data) const = 0;

    /** Reduce int data-set into `data` */
    virtual void reduce(view<int32_t> data) const = 0;

    /** Reduce long data-set into `data` */
    virtual void reduce(view<int64_t> data) const = 0;

    /** Finish reduction of all data if deferred */
    virtual void commit() const = 0;

    /** Returns a copy of `*this` created using `new` */
    virtual reducer *clone() { throw unsupported_operation(); }

    /** Destructor */
    virtual ~reducer() { }

    // Convenience functions

    void reduce(view<std::complex<double> > data) const {
        reduce(view<double>((double *)data.data(), 2 * data.size()));
    }
    void reduce(view<complex_op<double> > data) const {
        reduce(view<double>((double *)data.data(), 4 * data.size()));
    }
    void reduce(view<uint32_t> data) const {
        reduce(view<int32_t>((int32_t *)data.data(), data.size()));
    }
    void reduce(view<uint64_t> data) const {
        reduce(view<int64_t>((int64_t *)data.data(), data.size()));
    }
};

/**
 * Transformer instance.
 *
 * Note that multi-argument transformations are not supported.  Such
 * transformations must be implemented by first homogenizing the type and then
 * concatenating the argument vectors to a single argument.
 *
 * @see alps::alea::transform, alps::alea::join
 */
template <typename T>
struct transformer
{
    /** apply transformation */
    virtual column<T> operator() (const column<T> &in) const = 0;

    /** expected number of components of the input vector */
    virtual size_t in_size() const = 0;

    /** number of components of the returned vector */
    virtual size_t out_size() const = 0;

    /** Guarantee transformation to be linear (allows certain optimizations) */
    virtual bool is_linear() const { return false; }

    /** Destructor */
    virtual ~transformer() { }
};

}}
