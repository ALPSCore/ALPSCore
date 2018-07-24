/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
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
    bundle(size_t size, size_t target) : sum_(size), target_(target) { reset(); }

    /** Re-allocate and thus clear all accumulated data */
    void reset() { sum_.fill(0); count_ = 0; }

    bool is_full() { return count_ >= target_; }

    /** Number of components of the random vector (e.g., size of mean) */
    size_t size() const { return sum_.rows(); }

    size_t &target() { return target_; }

    const size_t &target() const { return target_; }

    size_t &count() { return count_; }

    size_t count() const { return count_; }

    column<T> &sum() { return sum_; }

    const column<T> &sum() const { return sum_; }

private:
    column<T> sum_;
    size_t target_, count_;
};

}}
