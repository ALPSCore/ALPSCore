/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_FIXED_CAPACITY_FWD_H
#define ALPS_FIXED_CAPACITY_FWD_H

#include <cstddef>

namespace alps {

namespace fixed_capacity {

// Forward declarations of checking policy classes
//   definitions are given in <alps/fixed_capacity/checking.h>

struct no_checking;
struct capacity_checking;
struct strict_checking;

} // namespace fixed_capacity

// Forward declarations of alps::fixed_capacity_[vector,deque]
//   definitions are given in <alps/fixed_capacity_vector.h> and
//   <alps/fixed_capacity_deque.h>, respectively

template<class T,
         std::size_t N,
         class CheckingPolicy = ::alps::fixed_capacity::no_checking>
class fixed_capacity_vector;
template<class T,
         std::size_t N,
         class CheckingPolicy = ::alps::fixed_capacity::no_checking>
class fixed_capacity_deque;

// Forward declaration of traits class alps::fixed_capacity_traits
//   definition is given in <alps/fixed_capacity_traits.h>

template<class C> struct fixed_capacity_traits;

} // namespace alps

#endif // ALPS_FIXED_CAPACITY_FWD_H
