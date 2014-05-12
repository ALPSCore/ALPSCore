/* 
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
*
 */

#include <stdexcept>
#include <boost/throw_exception.hpp>
#include <alps/random/parallel/keyword.hpp>

/// @file parallel/seed.hpp
/// 
/// This header provides default implementations of the parallel seeding functions.

#ifndef ALPS_RANDOM_PARALLEL_SEED_HPP
#define ALPS_RANDOM_PARALLEL_SEED_HPP

namespace alps { namespace random { namespace parallel {
  
  /// provides a default implementation of the default parallel seeding function
  /// by asssuming a named parameter seeding interface. It is implemented as
  /// @c prng.seed(stream_number=num, total_streams=total);
  ///
  /// Requirements: 0 <= @c num < @c total
  /// 
  /// @param prng the parallel random nubmber generator
  /// @param num the stream number
  /// @param total the total number of streams
  template <class PRNG>
  void seed(
          PRNG& prng
        , unsigned int num
        , unsigned int total
      )
  {
    prng.seed(stream_number=num, total_streams=total);
  }

  /// provides a convenience function for parallel seeding from a single global seed
  /// by asssuming a named parameter seeding interface. It is implemented as
  /// @c prng.seed(global_seed=s, stream_number=num, total_streams=total);
  ///
  /// Requirements: 0 <= @c num < @c total
  /// 
  /// @param prng the parallel random nubmber generator
  /// @param num the stream number
  /// @param total the total number of streams
  /// @param s the global seed
    template <class PRNG, class SeedType>
  void seed(
          PRNG& prng
        , unsigned int num
        , unsigned int total
        , SeedType const& s
      )
  {
    prng.seed(global_seed=s, stream_number=num, total_streams=total);
  }

  /// provides a default implementation of the parallel seeding from a pair iterators
  /// by asssuming a named parameter seeding interface. It is implemented as
  /// @c prng.seed(first, last, global_seed=s, stream_number=num, total_streams=total);
  ///
  /// Requirements: 0 <= @c num < @c total
  /// 
  /// @param prng the parallel random nubmber generator
  /// @param num the stream number
  /// @param total the total number of streams
  /// @param first an iterator to the beginning of a seed block
  /// @param last an iterator to the end of a seed block
  template <class PRNG, class Iterator>
  void seed(
          PRNG& prng
        , unsigned int num
        , unsigned int total
        , Iterator& first
        , Iterator const& last
      )
  {
   if(first == last)
      boost::throw_exception(std::invalid_argument("parallel_seed"));
    prng.seed(first,last,stream_number=num, total_streams=total);
  }


} } } // namespace alps::random::parallel

#endif // ALPS_RANDOM_PARALLEL_SEED_HPP
