/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#ifndef ALPS_RANDOM_PARALLEL_DETAIL_GET_PRIME_HPP
#define ALPS_RANDOM_PARALLEL_DETAIL_GET_PRIME_HPP

#include <boost/cstdint.hpp>

namespace alps { namespace random { namespace detail {

// get a prime number to be used as additive constant in a 64-bit LCG generator
    boost::uint64_t get_prime_64(unsigned int);

} } } // namespace alps::random::detail

#endif // ALPS_RANDOM_PARALLEL_DETAIL_GET_PRIME_HPP
