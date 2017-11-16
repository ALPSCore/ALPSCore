/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <stdexcept>
#include <complex>

#include <vector>
#include <Eigen/Dense>

#include <alps/alea/complex_op.hpp>

/** @namespace alps::alea
 *
 * Accumulators and results
 * ------------------------
 * Most accumulators (`mean_acc`) have a matching result class (`mean_result`).
 * To obtain a result from an accumulator, the accumulators provide both a
 * `result()` and a `finalize()` method, where
 *
 *  1. the `result()` method creates an intermediate result, which leaves the
 *     accumulator untouched and thus must involve a copy of the data, while
 *
 *  2. the `finalize()` method invalidates the accumulator and thus allows to
 *     repurpose its data as the simulation result.  The reset method then
 *     re-creates an empty accumulator with the same size.
 *
 * This can be represented by the following finite state machine:
 *
 *                     c'tor   _______________      _______________
 *                    ------->|               |    |               |  default
 *     result, <<        <<   |     empty     |    | uninitialized |   c'tor
 *      +-------+       +-----|_______________|    |_______________|<<-------
 *      |       |       |            | |
 *      |     __V_______V____  reset | | reset  ________________
 *      |    |               |--->---+ +---<---|                |
 *      +----|  accumulating |                 |     invalid    |
 *           |_______________|---------------->|________________|
 *                                finalize
 */
namespace alps { namespace alea {

using std::size_t;
using std::ptrdiff_t;

/** Estimator cannot add to sink as the sizes are mismatched */
struct size_mismatch : public std::exception { };

/** Estimator does not support this operation */
struct unsupported_operation : public std::exception { };

/** Accumulator has lost its data */
struct invalid_accumulator : public std::exception { };

/** Accumulator has lost its data */
struct uninitialized_accumulator : public std::exception { };

template <typename T>
struct traits;

/**
 * Data sink as a thin wrapper around a continuous array.
 *
 * Basically collects a pointer continuous array (vector, Eigen array, C
 * array etc.) together with its size.  This is sufficiently generic for the
 * virtual interface of `estimator`.
 */
template <typename T>
class sink
{
public:
    typedef T value_type;

public:
    /** Construct view on nothing */
    sink() : data_(NULL), size_(0) { }

    /** Construct view on data area with size */
    sink(T *data, size_t size) : data_(data), size_(size) { }

    /** Get pointer to zeroth element */
    T *data() { return data_; }

    /** Get pointer to zeroth element */
    const T *data() const { return data_; }

    /** Get size of data space */
    size_t size() const { return size_; }

private:
    T *data_;
    size_t size_;
};

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
    virtual void add_to(sink<T> out) const = 0;

    /** Returns a clone of the estimator (optional) */
    virtual computed *clone() { throw unsupported_operation(); }

    /** Destroy estimator */
    virtual ~computed() { }

    // FIXME: use a construction like this
    //   template <typename U>  U as();

    /** Allow for default conversions for convenience */
    operator std::vector<T>() const
    {
        std::vector<T> res(size(), 0);
        add_to(sink<T>(&res[0], res.size())); // TODO: data
        return res;
    }

    operator T() const
    {
        T res;
        add_to(sink<T>(&res, 1));
        return res;
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
    size_t count;

    /** Reductions will yield valid result on this instance */
    bool have_result;
};

/**
 * Perform sum-reduction with data from reducer source.
 *
 * The `reduce` methods take the data sink and add to it the data from the
 * reducers source (possibly by performing an MPI/OpenMP reduction or
 * gathering data from files, etc.).
 *
 * This facade allows us to abstract away the type of reduction, but most
 * importantly does not pull in a mandatory MPI dependency for the use of
 * the accumulators.
 */
struct reducer
{
    /** Set-up reduction operation */
    virtual reducer_setup get_setup() const = 0;

    /** Reduce double data-set into `data` */
    virtual void reduce(sink<double> data) = 0;

    /** Reduce long data-set into `data` */
    virtual void reduce(sink<long> data) = 0;

    /** Finish reduction of all data if deferred */
    virtual void commit() = 0;

    /** Destructor */
    virtual ~reducer() { }

    // Convenience functions

    void reduce(sink<std::complex<double> > data) {
        reduce(sink<double>((double *)data.data(), 2 * data.size()));
    }
    void reduce(sink<complex_op<double> > data) {
        reduce(sink<double>((double *)data.data(), 4 * data.size()));
    }
    void reduce(sink<unsigned long> data) {
        reduce(sink<long>((long *)data.data(), data.size()));
    }
};

/**
 * Foster the serialization of data to disk.
 */
struct serializer
{
    virtual void write(const std::string &key, const computed<double> &value) = 0;

    virtual void write(const std::string &key, const computed<std::complex<double> > &value) = 0;

    virtual void write(const std::string &key, const computed<complex_op<double> > &value) = 0;

    virtual void write(const std::string &key, const computed<long> &value) = 0;

    virtual ~serializer() { }
};

// TODO: refactor
template <typename InT, typename OutT>
struct transform
{
    virtual void operator() (sink<const InT> in, sink<OutT> out) = 0;
    virtual size_t out_size(size_t in_size) const = 0;
    virtual bool is_linear() const { return false; }
};

/** State flag for switching accumulator storages */
enum data_state { SUM, MEAN };


}}
