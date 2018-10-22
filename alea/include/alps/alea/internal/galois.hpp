/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

namespace alps { namespace alea { namespace internal {

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

}}}
