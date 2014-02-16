/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2003 by Synge Todo <wistaria@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/

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
