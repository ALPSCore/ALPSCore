/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_FIXED_CAPACITY_TRAITS_H
#define ALPS_FIXED_CAPACITY_TRAITS_H

#include <cstddef>
#include <queue>
#include <stack>
#include <boost/config.hpp>
#include <alps/fixed_capacity_fwd.h>

namespace alps {

// traits class alps::fixed_capacity_traits --------------------------------//

template<class C>
struct fixed_capacity_traits {
  BOOST_STATIC_CONSTANT(bool, capacity_is_fixed = false);
};

// specializations for fixed_capacity_[vector,deque]

template<class T, std::size_t N, class C>
struct fixed_capacity_traits<fixed_capacity_vector<T, N, C> > {
  BOOST_STATIC_CONSTANT(bool, capacity_is_fixed = true);
  BOOST_STATIC_CONSTANT(std::size_t, static_max_size = N);
};


template<class T, std::size_t N, class C>
struct fixed_capacity_traits<fixed_capacity_deque<T, N, C> > {
  BOOST_STATIC_CONSTANT(bool, capacity_is_fixed = true);
  BOOST_STATIC_CONSTANT(std::size_t, static_max_size = N);
};

// specializations for adaptors using fixed_capacity_[vector,deque] as
// a base container

template<class T, class C>
struct fixed_capacity_traits<std::stack<T, C> >
  : public fixed_capacity_traits<C> {};

template<class T, class C>
struct fixed_capacity_traits<std::queue<T, C> >
  : public fixed_capacity_traits<C> {};

template<class T, class C, class Cmp>
struct fixed_capacity_traits<std::priority_queue<T, C, Cmp> >
  : public fixed_capacity_traits<C> {};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class C>
const bool fixed_capacity_traits<C>::capacity_is_fixed;

template<class T, std::size_t N, class C>
const bool fixed_capacity_traits<fixed_capacity_vector<T, N, C> >::capacity_is_fixed;

template<class T, std::size_t N, class C>
const std::size_t fixed_capacity_traits<fixed_capacity_vector<T, N, C> >::static_max_size;

template<class T, std::size_t N, class C>
const bool fixed_capacity_traits<fixed_capacity_deque<T, N, C> >::capacity_is_fixed;

template<class T, std::size_t N, class C>
const std::size_t fixed_capacity_traits<fixed_capacity_deque<T, N, C> >::static_max_size;
#endif

} // namespace alps

#endif // ALPS_FIXED_CAPACITY_TRAITS_H
