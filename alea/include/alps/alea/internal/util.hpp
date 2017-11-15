/*
 * Set of auxiliary processing functions useful for implementations
 *
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */
#pragma once

#include <alps/alea/core.hpp>

namespace alps { namespace alea { namespace internal {

/** Ensures that the accumulator is initialized before doing stuff */
template <typename Acc>
inline void check_init(const Acc &acc)
{
    if (!acc.initialized())
        throw alps::alea::uninitialized_accumulator();
}

template <typename Acc>
inline void check_valid(const Acc &acc)
{
    if (!acc.valid()) {
        // First check if initialized
        check_init(acc);

        // If not, at least it is not valid
        throw alps::alea::invalid_accumulator();
    }
}

}}}
