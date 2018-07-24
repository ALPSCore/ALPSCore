/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#include <alps/alea/internal/galois.hpp>

namespace alps { namespace alea { namespace internal {

galois_hopper::galois_hopper(size_t size)
    : size_(size)
{
    if (size_ % 2 != 0)
        throw std::runtime_error("Number of batches must be even to allow "
                                 "for rebatching");
    reset();
}

void galois_hopper::reset(bool merge_mode)
{
    if (merge_mode) {
        level_ = 1;
        factor_ = 2;
        skip_ = 1;
    } else {
        level_ = 0;
        factor_ = 1;
        skip_ = 0;
    }
    cycle_ = 0;
    level_pos_ = 0;
    current_ = 0;
}

galois_hopper galois_hopper::operator++(int)
{
    galois_hopper save = *this;
    ++(*this);
    return save;
}

galois_hopper &galois_hopper::operator++()
{
    if (level_ == 0)
        advance_fill();
    else
        advance_galois();
    return *this;
}

void galois_hopper::advance_fill()
{
    assert(level_ == 0);
    ++current_;
    ++level_pos_;

    // we have filled all the elements, switch to Galois mode
    if (current_ == size_) {
        reset(true);
        ++cycle_;
    }
}

void galois_hopper::advance_galois()
{
    assert(level_ != 0);
    ++level_pos_;
    if (level_pos_ == size_/2) {
        ++level_;
        level_pos_ = 0;
        factor_ *= 2;
        skip_ *= 2;
    }
    current_ = (current_ + 2 * skip_) % (size_ + 1);
    assert(current_ != size_);

    // We have completed the cycle. Make sure skip does not overflow
    if (current_ == 0 && merge_into() == 1) {
        assert(level_pos_ == 0);
        ++cycle_;
    }
}

}}}
