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

#ifndef ALPS_FIXED_CAPACITY_DEQUE_H
#define ALPS_FIXED_CAPACITY_DEQUE_H

#include <boost/config.hpp>
#include <boost/static_assert.hpp>
#include <boost/type_traits/arithmetic_traits.hpp>
#include <algorithm>
#include <cstddef>
#include <iterator>
#include <stdexcept>

#include <alps/fixed_capacity_fwd.h>
#include <alps/fixed_capacity/uninitialized_array.h>
#include <alps/fixed_capacity/checking.h>

namespace alps {
  
namespace fixed_capacity {

// struct deque_iterator ----------------------------------------------------//

template<class T, std::size_t N, class Ref, class Ptr>
struct deque_iterator
{
  // types:
  typedef std::size_t                              size_type;
  typedef std::ptrdiff_t                           difference_type;
  typedef T                                        value_type;
  typedef Ref                                      reference;
  typedef Ptr                                      pointer;
  typedef deque_iterator<T, N, T&, T*>             iterator;
  typedef deque_iterator<T, N, const T&, const T*> const_iterator;
  typedef std::random_access_iterator_tag          iterator_category;

  // construct/copy/destroy:
  deque_iterator() : curr_(0), base_(0), first_(0) {}
  deque_iterator(pointer pos, pointer base, pointer first)
    : curr_(pos), base_(base), first_(first) {}
  deque_iterator(const iterator& x)
    : curr_(x.curr_), base_(x.base_), first_(x.first_) {}
  deque_iterator(const const_iterator& x)
    : curr_(x.curr_), base_(x.base_), first_(x.first_) {}
  // compiler-generated destructor is fine
   
  // dereference:
  reference operator*() const { return *curr_; }
  pointer operator->() const { return curr_; }
  reference operator[](difference_type n) const { return *(*this + n); }

  // difference:
  difference_type operator-(const deque_iterator& x) const {
    return pos_() - x.pos_();
  }

  deque_iterator& operator++() {
    ++curr_;
    if (curr_ == base_ + M) curr_ = base_;
    return *this;
  }
  deque_iterator operator++(int) {
    deque_iterator tmp = *this;
    ++*this;
    return tmp;
  }
  deque_iterator& operator--() {
    if (curr_ == base_) curr_ = base_ + M;
    --curr_;
    return *this;
  }
  deque_iterator operator--(int) {
    deque_iterator tmp = *this;
    --*this;
    return tmp;
  }
  deque_iterator& operator+=(difference_type n) {
    curr_ += n;
    if (curr_ >= base_ + M) {
      curr_ -= M;
    } else if (curr_ - base_ < 0) curr_ += M;
    return *this;
  }
  deque_iterator operator+(difference_type n) const {
    deque_iterator tmp = *this;
    tmp += n;
    return tmp;
  }
  deque_iterator& operator-=(difference_type n) { return *this += (-n); }
  deque_iterator operator-(difference_type n) const { return *this + (-n); }

  // comparison:
  bool operator==(const deque_iterator& x) const { return curr_ == x.curr_; }
  bool operator!=(const deque_iterator& x) const { return !(*this == x); }
  bool operator< (const deque_iterator& x) const {
    return pos_() < x.pos_();
  }
  bool operator> (const deque_iterator& x) const { return x < *this; }
  bool operator<=(const deque_iterator& x) const { return !(x < *this); }
  bool operator>=(const deque_iterator& x) const { return !(*this < x); }

  difference_type pos_() const {
    difference_type p = curr_ - first_;
    if (p < 0) p += M;
    return p;
  }

  BOOST_STATIC_ASSERT(N > 0);
  BOOST_STATIC_CONSTANT(std::size_t, M = N + 1); // size of array

  pointer curr_;
  pointer base_;
  pointer first_;

}; // deque_iterator

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<class T, std::size_t N, class Ref, class Ptr>
const std::size_t deque_iterator<T,N,Ref,Ptr>::M;
#endif


} // namespace fixed_capacity_deque


// class template fixed_capacity_deque --------------------------------------//

template<class T, std::size_t N, class CheckingPolicy>
class fixed_capacity_deque
{
private:
  typedef typename CheckingPolicy::BOOST_NESTED_TEMPLATE deque<N> checker;

public:
  // types:
  typedef std::size_t                              size_type;
  typedef std::ptrdiff_t                           difference_type;
  typedef T                                        value_type;
  typedef T&                                       reference;
  typedef const T&                                 const_reference;
  typedef fixed_capacity::deque_iterator<T, N, T&, T*>
                                                   iterator;
  typedef fixed_capacity::deque_iterator<T, N, const T&, const T*>
                                                   const_iterator;
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
    
  fixed_capacity_deque() { first_ = last_ = base(); }
  explicit fixed_capacity_deque(size_type n, const T& x = T()) {
    first_ = last_ = base();
    insert_n(end(), n, x);
  }
  template <class InputIterator>
  fixed_capacity_deque(InputIterator first, InputIterator last) {
    first_ = last_ = base();
    // dispatch depending on whether InputIterator is a integral type or not
    insert_dispatch(end(), first, last,
                    bool_type< ::boost::is_integral<InputIterator>::value>());
  }
  fixed_capacity_deque(const fixed_capacity_deque& x) {
    first_ = last_ = base();
    if (x.is_continuous()) {
      insert_dispatch(end(), (const T*)x.first_, (const T*)x.last_,
                      std::random_access_iterator_tag());
    } else {
      const T* x_base = x.base();
      insert_dispatch(end(), (const T*)x.first_, x_base + M,
                      std::random_access_iterator_tag());
      insert_dispatch(end(), x_base, (const T*)x.last_,
                      std::random_access_iterator_tag());
    }
  }
  ~fixed_capacity_deque() {
    if (is_continuous()) {
      destroy(first_, last_);
    } else {
      destroy(first_, top());
      destroy(base(), last_);
    }
  }
    
  // assignment:
  fixed_capacity_deque& operator=(const fixed_capacity_deque& x) {
    clear();
    if (x.is_continuous()) {
      insert_dispatch(end(), (const T*)x.first_, (const T*)x.last_,
                      std::random_access_iterator_tag());
    } else {
      const T* x_base = x.base();
      insert_dispatch(end(), (const T*)x.first_, x_base + M,
                      std::random_access_iterator_tag());
      insert_dispatch(end(), x_base, (const T*)x.last_,
                      std::random_access_iterator_tag());
    }
    return *this;
  }
  void assign(const T& x) {
    if (is_continuous()) {
      std::fill_n(first_, size(), x);
    } else {
      const size_type m = last_ - base();
      std::fill_n(first_, size() - m, x);
      std::fill_n(base(), m, x);
    }
  }
  void assign(size_type n, const T& x) {
    clear();
    insert_n(begin(), n, x);
  }
  template<class InputIterator>
  void assign(InputIterator first, InputIterator last) {
    clear();
    // dispatch depending on whether InputIterator is a integral type or not
    insert_dispatch(first_, first, last,
                    bool_type< ::boost::is_integral<InputIterator>::value>());
  }
  
  // iterators:
  iterator begin() { return iterator(first_, base(), first_); }
  const_iterator begin() const {
    return const_iterator(first_, base(), first_);
  }
  iterator end() { return iterator(last_, base(), first_); }
  const_iterator end() const {
    return const_iterator(last_, base(), first_);
  }
  reverse_iterator rbegin() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(end());
  }
  reverse_iterator rend() { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const {
    return const_reverse_iterator(begin());
  }
  
  // capacity:
  size_type size() const {
    difference_type s = last_ - first_;
    return (s >= 0) ? s : (s + M);
  }
  static size_type max_size() { return N; }
  void resize(size_type n, const T& c = T()) {
    checker::capacity_check(n);
    if (n > size()) {
      insert_n(end(), n - size(), c);
    } else {
      erase_n(begin() + n, size() - n);
    }
  }
  static size_type capacity() { return N; }
  bool empty() const { return first_ == last_; }
  void reserve(size_type n) const {
    checker::capacity_check(n);
  }
  
  // element access:
  reference operator[](size_type i) {
    return *(base() + position(i));
  }
  const_reference operator[](size_type i) const {
    return *(base() + position(i));
  }
  reference at(size_type i) { return operator[](i); }
  const_reference at(size_type i) const { return operator[](i); }
  reference front() {
    checker::range_check(size(), 0);
    return *first_;
  }
  const_reference front() const {
    checker::range_check(size(), 0);
    return *first_;
  }
  reference back() {
    checker::range_check(size(), 0);
    return (last_ == base()) ? *(base() + N) : *(last_ - 1);
  }
  const_reference back() const {
    checker::range_check(size(), 0);
    return (last_ == base()) ? *(base() + N) : *(last_ - 1);
  }
    
  // modifiers:
  void push_front(const T& x) {
    checker::capacity_check(size() + 1);
    if (first_ == base()) first_ += M;
    --first_;
    new ((void*)first_) T(x);
  }
  void push_back(const T& x) {
    checker::capacity_check(size() + 1);
    new ((void*)last_) T(x);
    ++last_;
    if (last_ == top()) last_ -= M;
  }
  void pop_front() {
    checker::range_check(size(), 0);
    destroy(first_);
    ++first_;
    if (first_ == top()) first_ -= M;
  }
  void pop_back() {
    checker::range_check(size(), 0);
    if (last_ == base()) last_ += M;
    --last_;
    destroy(last_);
  }
  iterator insert(iterator pos, const T& x) {
    checker::pointer_check(base(), first_, last_, &(*pos));
    if (pos == end()) {
      push_back(x);
      return pos;
    } else if (pos == begin()) {
      push_front(x);
      return begin();
    } else {
      return insert_n(pos, 1, x);
    }
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
      return pos;
    } else if (pos == begin()) {
      pop_front();
      return begin();
    } else {
      return erase_n(pos, 1);
    }
  }
  iterator erase(iterator first, iterator last) {
    return erase_n(first, last - first);
  }
  void swap(fixed_capacity_deque& x) {
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

protected:
  // pointer to uninitialized array
  T* base() { return data_.begin(); }
  const T* base() const { return data_.begin(); }
  T* top() { return data_.begin() + M; }
  const T* top() const { return data_.begin() + M; }

  // helper functions
  bool is_continuous() const { return last_ >= first_; }
  size_type position(size_type i) const {
    checker::range_check(size(), i);
    size_type p = (first_ - base()) + i;
    if (p >= M) p -= M;
    return p;
  }
  
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
                       std::random_access_iterator_tag);
  template<class InputIterator>
  void insert_dispatch(iterator pos, InputIterator first, InputIterator last,
                       std::input_iterator_tag) {
    // for general iterator: insert one by one
    if (pos == end()) {
      while (first != last) {
        push_back(*first);
        ++first;
      }
    } else if (pos == begin()) {
      while (first != last) {
        push_front(*first);
        ++first;
      }
    } else {
      while (first != last) {
        pos = insert_n(pos, 1, *first);
        ++pos;
        ++first;
      }
    }
  }

  iterator insert_n(iterator pos, size_type n, const T& x);
  iterator erase_n(iterator pos, size_type n);
  
  void destroy(T* pos) { pos->~T(); }
  void destroy(T* first, T* last) { while (first != last) (first++)->~T(); }
  
private:
  BOOST_STATIC_ASSERT(N > 0);
  BOOST_STATIC_CONSTANT(std::size_t, M = N + 1); // size of array

  T* first_; // pointer to the first element
  T* last_;  // pointer to next to the last element
  uninitialized_array<T, M> data_;

}; // fixed_capacity_deque

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template<class T, std::size_t N, class CheckingPolicy>
const std::size_t fixed_capacity_deque<T,N,CheckingPolicy>::M;
template<class T, std::size_t N, class CheckingPolicy>
const std::size_t fixed_capacity_deque<T,N,CheckingPolicy>::static_max_size;
#endif

template<class T, std::size_t N>
inline bool operator==(const fixed_capacity_deque<T, N>& x,
                       const fixed_capacity_deque<T, N>& y) {
  if (x.size() != y.size()) return false;
  return std::equal(x.begin(), x.end(), y.begin());
}
template<class T, std::size_t N>
inline bool operator< (const fixed_capacity_deque<T, N>& x,
                       const fixed_capacity_deque<T, N>& y) {
  return std::lexicographical_compare(x.begin(), x.end(), y.begin(), y.end());
}
template<class T, std::size_t N>
inline bool operator!=(const fixed_capacity_deque<T, N>& x,
                       const fixed_capacity_deque<T, N>& y) {
  return !(x == y);
}
template<class T, std::size_t N>
inline bool operator> (const fixed_capacity_deque<T, N>& x,
                       const fixed_capacity_deque<T, N>& y) {
  return y < x;
}
template<class T, std::size_t N>
inline bool operator<=(const fixed_capacity_deque<T, N>& x,
                       const fixed_capacity_deque<T, N>& y) {
  return !(y < x);
}
template<class T, std::size_t N>
inline bool operator>=(const fixed_capacity_deque<T, N>& x,
                       const fixed_capacity_deque<T, N>& y) {
  return !(x < y);
}
template<class T, std::size_t N>
inline void swap(fixed_capacity_deque<T, N>& x,
                 fixed_capacity_deque<T, N>& y) {
  x.swap(y);
}

} // namespace alps

#include <alps/fixed_capacity/deque_detail.h>

#endif // ALPS_FIXED_CAPACITY_DEQUE_H
