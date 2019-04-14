/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <cstddef>
#include <cassert>
#include <string>

namespace alps { namespace common {

using std::size_t;

/**
 * Data view as a thin wrapper around a continuous array.
 *
 * Basically collects a pointer continuous array together with its size.  Note
 * that the view does not own the `data` pointer.
 */
template <typename T>
class view
{
public:
    typedef T value_type;

public:
    /** Construct view on nothing */
    view() : data_(nullptr), size_(0) { }

    /** Construct view on data area with size */
    view(T *data, size_t size)
        : data_(data), size_(size)
    { }

    /** Get pointer to zeroth element */
    T *data() { return data_; }

    /** Get pointer to zeroth element */
    const T *data() const { return data_; }

    /** Return size */
    size_t size() const { return size_; }

private:
    T *data_;
    size_t size_;
};

/**
 * Data view as a thin wrapper around a continuous multi-dimensional array.
 *
 * Basically collects a pointer continuous array in ROW-MAJOR format (vector,
 * transposed Eigen array, C array etc.) together with its shape.  Note that
 * the view neither owns the `data` pointer nor the `shape` pointer.  Thus it
 * amends `view<T>` with shape information.
 */
template <typename T>
class ndview
    : public view<T>
{
public:
    typedef T value_type;

public:
    /** Construct view on nothing */
    ndview() : view<T>(), shape_(nullptr), ndim_(0) { }

    /** Construct view on data area with shape */
    ndview(T *data, const size_t *shape, size_t ndim)
        : view<T>(data, compute_size(shape, ndim))
        , shape_(shape)
        , ndim_(ndim)
    { }

    /** Construct view on data area with shape and size hint */
    ndview(T *data, size_t size, const size_t *shape, size_t ndim)
        : view<T>(data, size)
        , shape_(shape)
        , ndim_(ndim)
    {
        assert(size == compute_size(shape, ndim));
    }

    /** Get shape of data space */
    const size_t *shape() const { return shape_; }

    /** Get number of dimensions */
    size_t ndim() const { return ndim_; }

protected:
    static size_t compute_size(const size_t *shape, size_t ndim)
    {
        size_t result = 1;
        for (size_t d = 0; d != ndim; ++d)
            result *= shape[d];
        return result;
    }

private:
    const size_t *shape_;
    size_t ndim_;
};

}}
