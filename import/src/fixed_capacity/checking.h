/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_FIXED_CAPACITY_CHECKING_H
#define ALPS_FIXED_CAPACITY_CHECKING_H

#include <cstddef>
#include <stdexcept>

namespace alps {
  
namespace fixed_capacity {

// struct no_checking -------------------------------------------------------//

struct no_checking
{
  // for fixed_capacity_vector
  template<std::size_t N>
  struct vector {
    static void range_check(std::size_t, std::size_t) {}
    static void capacity_check(std::size_t) {}
    template<class T>
    static void pointer_check(const T*, const T*, const T*) {}
  };

  // for fixed_capacity_deque
  template<std::size_t N>
  struct deque {
    static void range_check(std::size_t, std::size_t) {}
    static void capacity_check(std::size_t) {}
    template<class T>
    static void pointer_check(const T*, const T*, const T*, const T*) {}
  };

}; // no_checking


// struct capacity_checking -------------------------------------------------//

struct capacity_checking
{
  // for fixed_capacity_vector
  template<std::size_t N>
  struct vector {
    static void range_check(std::size_t, std::size_t) {}
    static void capacity_check(std::size_t n) {
      if (n > N) throw std::range_error("fixed_capacity_vector");
    }
    template<class T>
    static void pointer_check(const T*, const T*, const T*) {}
  };

  // for fixed_capacity_deque
  template<std::size_t N>
  struct deque {
    static void range_check(std::size_t, std::size_t) {}
    static void capacity_check(std::size_t n) {
      if (n > N) throw std::range_error("fixed_capacity_deque");
    }
    template<class T>
    static void pointer_check(const T*, const T*, const T*, const T*) {}
  };

}; // capacity_checking


// struct strict_checking ---------------------------------------------------//

struct strict_checking
{
  // for fixed_capacity_vector
  template<std::size_t N>
  struct vector {
    static void range_check(std::size_t s, std::size_t i) {
      if (i >= s) throw std::range_error("fixed_capacity_vector");
    }
    static void capacity_check(std::size_t n) {
      if (n > N) throw std::range_error("fixed_capacity_vector");
    }
    template<class T>
    static void pointer_check(const T* base, const T* last, const T* ptr) {
      if (ptr < base || ptr > last)
        throw std::range_error("fixed_capacity_vector");
    }
  };

  // for fixed_capacity_deque
  template<std::size_t N>
  struct deque {
    static void range_check(std::size_t s, std::size_t i) {
      if (i >= s) throw std::range_error("fixed_capacity_deque");
    }
    static void capacity_check(std::size_t n) {
      if (n > N) throw std::range_error("fixed_capacity_deque");
    }
    template<class T>
    static void pointer_check(const T* base, const T* first, const T* last,
                              const T* ptr) {
      if (last - first >= 0) {
        if (ptr < first || ptr > last) {
          throw std::range_error("fixed_capacity_deque");
        }
      } else {
        if (ptr < base || (ptr > last && ptr < first) || ptr > base + N) {
          throw std::range_error("fixed_capacity_deque");
        }
      }
    }
  };

}; // strict_checking

} // namespace fixed_capacity

} // namespace alps

#endif // ALPS_FIXED_CAPACITY_CHECKING_H
