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

#ifndef ALPS_FIXED_CAPACITY_VECTOR_H
#define ALPS_FIXED_CAPACITY_VECTOR_H

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/arithmetic_traits.hpp>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <memory>

#include <alps/fixed_capacity_fwd.h>
#include <alps/fixed_capacity/uninitialized_array.h>
#include <alps/fixed_capacity/checking.h>

namespace alps {

// class template fixed_capacity_vector -------------------------------------//

template<class T, std::size_t N, class CheckingPolicy>
class fixed_capacity_vector
{
private:
  typedef typename CheckingPolicy::BOOST_NESTED_TEMPLATE vector<N> checker;

public:
  // types:
  typedef std::size_t                              size_type;
  typedef std::ptrdiff_t                           difference_type;
  typedef T                                        value_type;
  typedef T&                                       reference;
  typedef const T&                                 const_reference;
  typedef T*                                       iterator;
  typedef const T*                                 const_iterator;
#if !defined(BOOST_NO_TEMPLATE_PARTIAL_SPECIALIZATION) && \
  !defined(BOOST_MSVC_STD_ITERATOR)
  typedef std::reverse_iterator<iterator>          reverse_iterator;
  typedef std::reverse_iterator<const_iterator>    const_reverse_iterator;
#else
  // workaround for broken reverse_iterator implementations
  typedef std::reverse_iterator<iterator, T>       reverse_iterator;
  typedef std::reverse_iterator<const_iterator, T> const_reverse_iterator;
#endif
  
  BOOST_STATIC_CONSTANT(size_type, static_max_size = N);
  
  // construct/copy/destroy:
  fixed_capacity_vector() { last_ = base(); }
  explicit fixed_capacity_vector(size_type n, const T& x = T()) {
    last_ = base();
    insert_n(last_, n, x);
  }
  template <class InputIterator>
  fixed_capacity_vector(InputIterator first, InputIterator last) {
    last_ = base();
    // dispatch depending on whether InputIterator is a integral type or not
    insert_dispatch(last_, first, last,
                    bool_type< ::boost::is_integral<InputIterator>::value>());
  }
  fixed_capacity_vector(const fixed_capacity_vector& x) {
    last_ = base();
    insert_dispatch(last_, x.begin(), x.end(),
                    std::random_access_iterator_tag());
  }
  ~fixed_capacity_vector() { destroy(base(), last_); }
  
  // assignment:
  fixed_capacity_vector& operator=(const fixed_capacity_vector& x) {
    clear();
    insert_dispatch(last_, x.begin(), x.end(),
                    std::random_access_iterator_tag());
    return *this;
  }
  void assign(const T& x) { std::fill_n(base(), size(), x); }
  void assign(size_type n, const T& x) {
    clear();
    insert_n(last_, n, x);
  }
  template<class InputIterator>
  void assign(InputIterator first, InputIterator last) {
    // dispatch depending on whether InputIterator is a integral type or not
    clear();
    insert_dispatch(base(), first, last,
                    bool_type< ::boost::is_integral<InputIterator>::value>());
  }
  
  // iterators:
  iterator begin() { return base(); }
  const_iterator begin() const { return base(); }
  iterator end() { return last_; }
  const_iterator end() const { return last_; }
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  
  // capacity:
  size_type size() const { return last_ - base(); }
  static size_type max_size() { return N; }
  void resize(size_type n, const T& c = T()) {
    checker::capacity_check(n);
    if (n > size()) {
      insert_n(last_, n - size(), c);
    } if (n < size()) {
      erase_n(base() + n, size() - n);
    }
  }
  static size_type capacity() { return N; }
  bool empty() const { return base() == last_; }
  void reserve(size_type n) const {
    checker::capacity_check(n);
  }
  
  // element access:
  reference operator[](size_type i) {
    checker::range_check(size(), i);
    return *(base() + i);
  }
  const_reference operator[](size_type i) const {
    checker::range_check(size(), i);
    return *(base() + i);
  }
  reference at(size_type i) { return operator[](i); }
  const_reference at(size_type i) const { return operator[](i); }
  reference front() {
    checker::range_check(size(), 0);
    return *base();
  }
  const_reference front() const {
    checker::range_check(size(), 0);
    return *base();
  }
  reference back() {
    checker::range_check(size(), 0);
    return *(last_ - 1);
  }
  const_reference back() const {
    checker::range_check(size(), 0);
    return *(last_ - 1);
  }
  
  // modifiers:
  void push_back(const T& x) {
    checker::capacity_check(size() + 1);
    new ((void*)last_) T(x);
    ++last_;
  }
  void pop_back() {
    checker::range_check(size(), 0);
    destroy(--last_);
  }
  iterator insert(iterator pos, const T& x) {
    checker::pointer_check(base(), last_, pos);
    if (pos == last_) {
      push_back(x);
    } else {
      checker::capacity_check(size() + 1);
      new ((void*)last_) T(*(last_ - 1));
      std::copy_backward(pos, last_ - 1, last_);
      *pos = x;
      ++last_;
    }
    return pos;
  }
  void insert(iterator pos, size_type n, const T& x) { insert_n(pos, n, x); }
  template<class InputIterator>
  void insert(iterator pos, InputIterator first, InputIterator last) {
    // dispatch depending on whether InputIterator is a integral type or not
    insert_dispatch(pos, first, last,
                    bool_type< ::boost::is_integral<InputIterator>::value>());
  }
  iterator erase(iterator pos) {
    if (pos == end()) {
      pop_back();
    } else {
      erase_n(pos, 1);
    }
    return pos;
  }
  iterator erase(iterator first, iterator last) {
    erase_n(first, last - first);
    return first;
  }
  void swap(fixed_capacity_vector& x) {
    if (size() <= x.size()) {
      size_type d = x.size() - size();
      std::swap_ranges(begin(), end(), x.begin());
      insert(end(), x.begin() + size(), x.end());
      x.erase(x.end() - d, x.end());
    } else {
      size_type d = size() - x.size();
      std::swap_ranges(x.begin(), x.end(), begin());
      x.insert(x.end(), begin() + x.size(), end());
      erase(end() - d, end());
    }
  }
  void clear() { erase(begin(), end()); }
  
  // direct access to data
  const T* data() const { return base(); }
  
protected:
  // pointer to uninitialized array
  T* base() { return data_.begin(); }
  const T* base() const { return data_.begin(); }

  // helper class for dispatching
  template<bool B> struct bool_type {};

  // helper functions for insertion
  template<class U>
  void insert_dispatch(iterator pos, size_type n, U x, bool_type<true>) {
    // for integral-type second argument
    insert_n(pos, n, x);
  }
  template<class InputIterator>
  void insert_dispatch(iterator pos, InputIterator first, InputIterator last,
                       bool_type<false>) {
    // for interator-type second argument: dispatch depending on
    // whether InputIterator is a random access iterator or not
    insert_dispatch(pos, first, last,
                    typename std::iterator_traits<InputIterator>::iterator_category());
  }
  template<class InputIterator>
  void insert_dispatch(iterator pos, InputIterator first, InputIterator last,
                       std::random_access_iterator_tag) {
    // for random access iterator
    const size_type n = last - first;
    checker::capacity_check(size() + n);
    checker::pointer_check(base(), last_, pos);
    const size_type m = last_ - pos;
    if (m == 0) {
      std::uninitialized_copy(first, last, last_);
    } else {
      if (n < m) {
        std::uninitialized_copy(last_ - n, last_, last_);
        std::copy_backward(pos, last_ - n, last_);
        std::copy(first, last, pos);
      } else {
        std::uninitialized_copy(pos, last_, last_ + n - m);
        std::copy(first, first + m, pos);
        std::uninitialized_copy(first + m, last, last_);
      }
    }
    last_ += n;
  }
  template<class InputIterator>
  void insert_dispatch(iterator pos, InputIterator first, InputIterator last,
                       std::input_iterator_tag) {
    // for general iterator: insert one by one
    if (pos == end()) {
      while (first != last) {
        push_back(*first);
        ++first;
      }
    } else {
      while (first != last) {
        insert_n(pos, 1, *first);
        ++pos;
        ++first;
      }
    }
  }

  void insert_n(T* pos, size_type n, const T& x) {
    checker::capacity_check(size() + n);
    checker::pointer_check(base(), last_, pos);
    const size_type m = last_ - pos;
    if (m == 0) {
      std::uninitialized_fill(last_, last_ + n, x);
    } else {
      if (n < m) {
        std::uninitialized_copy(last_ - n, last_, last_);
        std::copy_backward(pos, last_ - n, last_);
        std::fill(pos, pos + n, x);
      } else {
        std::uninitialized_copy(pos, last_, pos + n);
        std::fill(pos, last_, x);
        std::uninitialized_fill(last_, pos + n, x);
      }
    }
    last_ += n;
  }
  void erase_n(T* pos, size_type n) {
    checker::capacity_check(size() - n);
    checker::pointer_check(base(), last_, pos);
    checker::pointer_check(base(), last_, pos + n);
    last_ = std::copy(pos + n, last_, pos);
    destroy(last_, last_ + n);
  }

  void destroy(T* pos) { pos->~T(); }
  void destroy(T* first, T* last) { while (first != last) (first++)->~T(); }
  
private:
  BOOST_STATIC_ASSERT(N > 0);

  T* last_; // pointer to next to the last element
  uninitialized_array<T, N> data_;

}; // fixed_capacity_vector
  

// global functions ---------------------------------------------------------//
  
template<class T, std::size_t N>
inline bool operator==(const fixed_capacity_vector<T, N>& x,
                       const fixed_capacity_vector<T, N>& y) {
  if (x.size() != y.size()) return false;
  return std::equal(x.begin(), x.end(), y.begin());
}
template<class T, std::size_t N>
inline bool operator< (const fixed_capacity_vector<T, N>& x,
                       const fixed_capacity_vector<T, N>& y) {
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}
template<class T, std::size_t N>
inline bool operator!=(const fixed_capacity_vector<T, N>& x,
                       const fixed_capacity_vector<T, N>& y) {
  return !(x == y);
}
template<class T, std::size_t N>
inline bool operator> (const fixed_capacity_vector<T, N>& x,
                       const fixed_capacity_vector<T, N>& y) {
  return y < x;
}
template<class T, std::size_t N>
inline bool operator<=(const fixed_capacity_vector<T, N>& x,
                       const fixed_capacity_vector<T, N>& y) {
  return !(y < x);
}
template<class T, std::size_t N>
inline bool operator>=(const fixed_capacity_vector<T, N>& x,
                       const fixed_capacity_vector<T, N>& y) {
  return !(x < y);
}
template<class T, std::size_t N>
inline void swap(fixed_capacity_vector<T, N>& x,
                 fixed_capacity_vector<T, N>& y) {
  x.swap(y);
}

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<class T, std::size_t N, class CheckingPolicy>
const typename fixed_capacity_vector<T,N,CheckingPolicy>::size_type
fixed_capacity_vector<T,N,CheckingPolicy>::static_max_size;
#endif

} // namespace alps

#endif // ALPS_FIXED_CAPACITY_VECTOR_H
