/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_MULTI_ARRAY_OPERATORS_HPP
#define ALPS_MULTI_ARRAY_OPERATORS_HPP

#include <alps/multi_array/multi_array.hpp>

namespace alps{

  template <class T, std::size_t D, class Allocator>
  multi_array<T,D,Allocator> operator-(multi_array<T,D,Allocator> arg)
  {
    std::transform(arg.data(), arg.data() + arg.num_elements(), arg.data(), std::negate<T>());
    return arg;
  }

  template <class T, std::size_t D, class Allocator>
  multi_array<T,D,Allocator> operator+(multi_array<T,D,Allocator> a, const multi_array<T,D,Allocator>& b)
  {
    a += b;
    return a;
  }

  template <class T, std::size_t D, class Allocator>
  multi_array<T,D,Allocator> operator-(multi_array<T,D,Allocator> a, const multi_array<T,D,Allocator>& b)
  {
    a -= b;
    return a;
  }

  template <class T, std::size_t D, class Allocator>
  multi_array<T,D,Allocator> operator*(multi_array<T,D,Allocator> a, const multi_array<T,D,Allocator>& b)
  {
    a *= b;
    return a;
  }

  template <class T, std::size_t D, class Allocator>
  multi_array<T,D,Allocator> operator/(multi_array<T,D,Allocator> a, const multi_array<T,D,Allocator>& b)
  {
    a /= b;
    return a;
  }

  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator+(const T2& b, multi_array<T1,D,Allocator> a)
  {
    a += T1(b);
    return a;
  }
  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator+(multi_array<T1,D,Allocator> a, const T2& b)
  {
    a += T1(b);
    return a;
  }

  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator-(const T2& b, multi_array<T1,D,Allocator> a)
  {
    a -= T1(b);
    return a;
  }
  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator-(multi_array<T1,D,Allocator> a, const T2& b)
  {
    a -= T1(b);
    return -a;
  }

  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator*(const T2& b, multi_array<T1,D,Allocator> a)
  {
    a *= T1(b);
    return a;
  }

  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator*(multi_array<T1,D,Allocator> a, const T2& b)
  {
    a *= T1(b);
    return a;
  }

  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T1,D,Allocator> operator/(multi_array<T1,D,Allocator> a, const T2& b)
  {
    a /= T1(b);
    return a;
  }
  template <class T1, class T2, std::size_t D, class Allocator>
  multi_array<T2,D,Allocator> operator/(T2 const & scalar, multi_array<T1,D,Allocator> array)
  {
    std::transform(array.data(), array.data() + array.num_elements(), array.data(), std::bind1st(std::divides<T1>(),T1(scalar)));
    return array;
  }

}//namespace alps

#endif // ALPS_MULTI_ARRAY_OPERATORS_HPP
