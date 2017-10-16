/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>
#include <alps/alea/util.hpp>

namespace alps { namespace alea {

/**
 * Bundle of measurements.
 *
 * Some accumulator, particularly the variance, are more efficient working in
 * "bundled" mode, where n measurement are bundled or batched together.
 */
template <typename T>
class bundle
{
public:
    bundle(size_t size, size_t cap) : sum_(size), capacity_(cap) { reset(); }

    void reset() { sum_.fill(0); count_ = 0; }

    bool is_full() { assert(count_ <= capacity_); return count_ == capacity_; }

    size_t size() const { return sum_.rows(); }

    size_t &capacity() { return capacity_; }

    const size_t &capacity() const { return capacity_; }

    size_t &count() { return count_; }

    const size_t &count() const { return count_; }

    column<T> &sum() { return sum_; }

    const column<T> &sum() const { return sum_; }

private:
    column<T> sum_;
    size_t capacity_, count_;
};

}}
