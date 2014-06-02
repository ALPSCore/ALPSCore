/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_RANDOM_PARALLEL_DETAIL_GET_PRIME_HPP
#define ALPS_RANDOM_PARALLEL_DETAIL_GET_PRIME_HPP

#include <boost/cstdint.hpp>

namespace alps { namespace random { namespace detail {

// get a prime number to be used as additive constant in a 64-bit LCG generator
    boost::uint64_t get_prime_64(unsigned int);

} } } // namespace alps::random::detail

#endif // ALPS_RANDOM_PARALLEL_DETAIL_GET_PRIME_HPP
