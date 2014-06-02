/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#include <cmath>
#include <boost/random.hpp>
#include <cstdlib>
#include <iostream>
#include <vector>

template<class T>
struct non_pod {
  BOOST_STATIC_CONSTANT(int, magic = -1190884);

  non_pod() {
    if (init_ == magic) throw std::logic_error("non_pod");
    init_ = magic;
    data_ = 0;
  }
  non_pod(const non_pod& x) {
    if (init_ == magic) throw std::logic_error("non_pod");
    init_ = magic;
    data_ = x.data_;
  }
  non_pod(const T& x) {
    if (init_ == magic) throw std::logic_error("non_pod");
    init_ = magic;
    data_ = x;
  }
  ~non_pod() {
    int init = init_;
    T data = data_;
    init_ = 0;
    data_ = -1.0;
    if (init != magic) throw std::logic_error("non_pod");
    if (data < 0) std::cerr << "warning\n";
  }
  non_pod& operator=(const non_pod& x) {
    if (init_ != magic) throw std::logic_error("non_pod");
    data_ = x.data_;
    return *this;
  }
  non_pod& operator=(const T& x) {
    if (init_ != magic) throw std::logic_error("non_pod");
    data_ = x;
    return *this;
  }
  
  bool operator==(const non_pod& x) const {
    if (init_ != magic || x.init_ != magic) throw std::logic_error("non_pod");
    return data_ == x.data_;
  }
  bool operator!=(const non_pod& x) const { return !operator==(x); }
  
  friend bool operator==(T x, const non_pod& y) { return y == x; }
  friend bool operator!=(T x, const non_pod& y) { return y != x; }
  friend std::ostream& operator<<(std::ostream& os, const non_pod& x) {
    os << x.data_;
    return os;
  }
  
  T data_;
  int init_;
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
template <class T>
const int non_pod<T>::magic;
#endif

template<class S, class T, class U>
bool check(const S& s, const T& t, const U& u) {
  bool check = true;
  if (s.size() != t.size() || s.size() != u.size()) return false;
  typename T::const_iterator t_iter = t.begin();
  typename U::const_iterator u_iter = u.begin();
  for (typename S::const_iterator iter = s.begin(); iter != s.end();
       ++iter) {
#ifdef VERBOSE    
    std::cout << "(" << *iter << "," << *t_iter << "," << *u_iter << ") ";
#endif
    if (*iter != *t_iter || *iter != *u_iter) check = false;
    ++t_iter;
    ++u_iter;
  }
#ifdef VERBOSE    
  std::cout << std::endl;
#endif
  return check;
}


template<class Vec, class RNG>
void make_array(Vec& vec, RNG& rng, std::size_t n) {
  vec.clear();
  for (std::size_t i = 0; i < n; ++i) vec.push_back(rng());
}

template<class S, class T, class U>
void test_main(std::size_t m, std::size_t n) {
  S s;
  T t;
  U u;
    
  std::vector<double> v;
  boost::lagged_fibonacci607 rng;
  
  for (std::size_t i = 0; i < m; ++i) {
#ifdef VERBOSE
    std::cout << i << ' ';
#endif
    double r = rng();
    if (r < 0.1) {
      // clear
#ifdef VERBOSE
      std::cout << "clear\n";
#endif
      s.clear();
      t.clear();
      u.clear();
    } else if (r < 0.2) {
      // push_back
      double d = rng();
      if (s.size() < n) {
#ifdef VERBOSE
        std::cout << "push_back " << d << std::endl;
#endif
        s.push_back(d);
        t.push_back(d);
        u.push_back(d);
      } else {
#ifdef VERBOSE
        std::cout << std::endl;
#endif
      }
#ifdef DEQUE
    } else if (r < 0.25) {
      // push_front
      double d = rng();
      if (s.size() < n) {
#ifdef VERBOSE
        std::cout << "push_front " << d << std::endl;
#endif
        s.push_front(d);
        t.push_front(d);
        u.push_front(d);
      } else {
#ifdef VERBOSE
        std::cout << std::endl;
#endif
      }
#endif // DEQUE
    } else if (r < 0.3) {
      // pop_back
      if (!s.empty()) {
#ifdef VERBOSE
        std::cout << "pop_back\n";
#endif
        s.pop_back();
        t.pop_back();
        u.pop_back();
      } else {
#ifdef VERBOSE
        std::cout << std::endl;
#endif
      }
#ifdef DEQUE      
    } else if (r < 0.35) {
      // pop_front
      if (!s.empty()) {
#ifdef VERBOSE
        std::cout << "pop_front\n";
#endif
        s.pop_front();
        t.pop_front();
        u.pop_front();
      } else {
#ifdef VERBOSE
        std::cout << std::endl;
#endif
      }
#endif // DEQUE
    } else if (r < 0.4) {
      // resize
      double d = rng();
      std::size_t p = int(n * rng());
#ifdef VERBOSE
      std::cout << "resize " << p << ' ' << d << std::endl;
#endif
      s.resize(p, d);
      t.resize(p, d);
      u.resize(p, d);
    } else if (r < 0.7) {
      // insert
      std::size_t p = int(s.size() * rng());
      std::size_t x = int((n - s.size()) * rng());
      if (rng() < 0.5) {
        double d = rng();
#ifdef VERBOSE
        std::cout << "insert " << p << ' ' << x << ' ' << d << std::endl;
#endif
        if (x != 1) {
          s.insert(s.begin() + p, x, d);
          t.insert(t.begin() + p, x, d);
          u.insert(u.begin() + p, x, d);
        } else {
          s.insert(s.begin() + p, d);
          t.insert(t.begin() + p, d);
          u.insert(u.begin() + p, d);
        }
      } else {
#if !(__GNUC__ == 3 && __GNUC_MINOR__ == 1)
        // gcc-3.1 has a bug in std::uninitialized_copy,
        // so just skip this test.
#ifdef VERBOSE
        std::cout << "insert sequence\n";
#endif
        make_array(v, rng, x);
        s.insert(s.begin() + p, v.begin(), v.end());
        t.insert(t.begin() + p, v.begin(), v.end());
        u.insert(u.begin() + p, v.begin(), v.end());
#endif // !(__GNUC__ == 3 && __GNUC_MINOR__ == 1)
      }
    } else {
      // erase
      std::size_t p = int(s.size() * rng());
      std::size_t x = p + int((s.size() - p) * rng());
#ifdef VERBOSE
      std::cout << "erase " << p << ' ' << x << std::endl;
#endif
      if (x != 1) {
        s.erase(s.begin() + p, s.begin() + x);
        t.erase(t.begin() + p, t.begin() + x);
        u.erase(u.begin() + p, u.begin() + x);
      } else {
        s.erase(s.begin() + p);
        t.erase(t.begin() + p);
        u.erase(u.begin() + p);
      }
    } 
    
    if (!check(s, t, u)) {
      std::cout << "Error occured!\n";
      std::exit(-1);
    }
  }
  
  return;
}
