/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_PERMUTATION_H
#define PARAPACK_PERMUTATION_H

#include <algorithm>
#include <boost/random/uniform_real.hpp>
#include <iterator>

namespace alps {

template<class RandomAccessIter, class RandomNumberGenerator>
void random_shuffle(RandomAccessIter first, RandomAccessIter last, RandomNumberGenerator& rng) {
  using std::iter_swap;
  for (typename std::iterator_traits<RandomAccessIter>::difference_type
         n = last - first; n > 1; ++first, --n)
    iter_swap(first, first + (int)(n * rng()));
}

} // end namespace alps

#endif // PARAPACK_PERMUTATION_H
