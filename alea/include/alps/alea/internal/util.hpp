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

template <typename Acc>
inline void check_valid(const Acc &acc)
{
    if (!acc.valid())
        throw alps::alea::finalized_accumulator();
}

}}}
