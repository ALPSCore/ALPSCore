/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2006-2013 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_RANDOM_CHOICE_HPP
#define ALPS_RANDOM_CHOICE_HPP

#include <boost/foreach.hpp>
#include <boost/static_assert.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>
#include <iostream>

namespace alps {

namespace detail {

template<typename WVEC, typename CutoffType, typename IndexType>
inline bool check_table(WVEC const& weights,
  std::vector<std::pair<CutoffType, IndexType> > const& table, double tol = 1.0e-10) {
  std::size_t n = weights.size();
  std::size_t m = table.size();
  bool r = true;
  tol *= n;
  double norm = m / std::accumulate(weights.begin(), weights.end(), double(0));
  double nm = 1;
  if (boost::is_integral<CutoffType>::value) nm /= std::numeric_limits<CutoffType>::max();
  for (IndexType i = 0; i < n; ++i) {
    double p = nm * table[i].first;
    for (IndexType j = 0; j < m; ++j)
      if (table[j].second == i) p += (1.0 - nm * table[j].first);
    r &= std::abs(p - norm * weights[i]) < tol;
  }
  return r;
}

// Initialization routine with complexity O(N)
template<typename WVEC, typename CutoffType, typename IndexType>
inline void fill_ft2009(WVEC const& weights, std::vector<std::pair<CutoffType, IndexType> >& table,
  typename boost::enable_if<boost::is_float<CutoffType> >::type* = 0,
  typename boost::enable_if<boost::is_integral<IndexType> >::type* = 0) {
  
  if (weights.size() == 0)
    boost::throw_exception(std::invalid_argument("fill_ft2009"));
  std::size_t n = weights.size();
  CutoffType norm = CutoffType(0);
  BOOST_FOREACH(CutoffType w, weights) {
    if (w < CutoffType(0))
      boost::throw_exception(std::invalid_argument("fill_ft2009"));
    norm += w;
  }
  if (norm <= CutoffType(0))
    boost::throw_exception(std::invalid_argument("fill_ft2009"));
  norm = n / norm;

  // Initialize arrays.  We will reorder the elements in `array', so
  // that all the negative elements precede the positive ones.
  table.resize(n);
  std::vector<std::pair<CutoffType, IndexType> > array(n);
  typename std::vector<std::pair<CutoffType, IndexType> >::iterator neg_p = array.begin();
  typename std::vector<std::pair<CutoffType, IndexType> >::iterator pos_p = array.end();
  for (std::size_t i = 0; i < n; ++i) {
    CutoffType b = norm * weights[i] - CutoffType(1);
    if (b < CutoffType(0)) {
      *neg_p = std::make_pair(b, i);
      ++neg_p;
    } else {
      --pos_p;
      *pos_p = std::make_pair(b, i);
    }
  }

  // Note: at this point `pos_p' is pointing the first non-negative element in the array.

  // Assign alias and cutoff values
  for (neg_p = array.begin(); neg_p != array.end(); ++neg_p) {
    if (pos_p != array.end()) {
      table[neg_p->second] = std::make_pair(CutoffType(1) + neg_p->first, pos_p->second);
      pos_p->first += neg_p->first;
      if (pos_p->first <= CutoffType(0)) ++pos_p;
    } else {
      table[neg_p->second] = std::make_pair(CutoffType(1), neg_p->second);
    }
  }
}
  
template<typename WVEC, typename CutoffType, typename IndexType>
inline void fill_ft2009(WVEC const& weights, std::vector<std::pair<CutoffType, IndexType> >& table,
  typename boost::enable_if<boost::is_integral<CutoffType> >::type* = 0,
  typename boost::enable_if<boost::is_integral<IndexType> >::type* = 0) {
  if (weights.size() == 0)
    boost::throw_exception(std::range_error("fill_ft2009"));
  std::size_t n = weights.size();
  std::size_t m = 2;
  while (m < n) m <<= 1;

  double norm = 0;
  BOOST_FOREACH(double w, weights) {
    if (w < 0)
      boost::throw_exception(std::invalid_argument("fill_ft2009"));
    norm += w;
  }
  if (norm <= 0)
    boost::throw_exception(std::invalid_argument("fill_ft2009"));
  norm = m / norm;

  // Initialize arrays.  We will reorder the elements in `array', so
  // that all the negative elements precede the positive ones.
  table.resize(m);
  std::vector<std::pair<double, IndexType> > array(m);
  typename std::vector<std::pair<double, IndexType> >::iterator neg_p = array.begin();
  typename std::vector<std::pair<double, IndexType> >::iterator pos_p = array.end();
  for (std::size_t i = 0; i < m; ++i) {
    double b = norm * (i < n ? weights[i] : 0) - 1;
    if (b < 0) {
      *neg_p = std::make_pair(b, i);
      ++neg_p;
    } else {
      --pos_p;
      *pos_p = std::make_pair(b, i);
    }
  }

  // Note: now `pos_p' is pointing the first non-negative element in the array.

  // Assign alias and cutoff values
  double nm = std::numeric_limits<CutoffType>::max();
  for (neg_p = array.begin(); neg_p != array.end(); ++neg_p) {
    if (pos_p != array.end()) {
      table[neg_p->second] = std::make_pair(CutoffType(nm * (1 + neg_p->first)), pos_p->second);
      pos_p->first += neg_p->first;
      if (pos_p->first <= 0) ++pos_p;
    } else {
      table[neg_p->second] = std::make_pair(CutoffType(nm), neg_p->second);
    }
  }
}

// Original O(N^2) initialization routine given in A. W. Walker, ACM
// Trans. Math. Software, 3, 253 (1977).
template<typename WVEC, typename CutoffType, typename IndexType>
inline void fill_walker1977(WVEC const& weights,
  std::vector<std::pair<CutoffType, IndexType> >& table, CutoffType tol = 1.0e-10,
  typename boost::enable_if<boost::is_float<CutoffType> >::type* = 0,
  typename boost::enable_if<boost::is_integral<IndexType> >::type* = 0) {

  if (weights.size() == 0)
    boost::throw_exception(std::invalid_argument("fill_walker1977"));
  std::size_t n = weights.size();
  CutoffType norm = CutoffType(0);
  BOOST_FOREACH(CutoffType w, weights) {
    if (w < CutoffType(0))
      boost::throw_exception(std::invalid_argument("fill_walker1977"));
    norm += w;
  }
  if (norm <= CutoffType(0))
    boost::throw_exception(std::invalid_argument("fill_walker1977"));
  norm = n / norm;

  // Initialize arrays.
  table.resize(n);
  std::vector<CutoffType> b(n);
  for (std::size_t i = 0; i < n; ++i) {
    table[i] = std::make_pair(CutoffType(1), i);
    b[i] = norm * weights[i] - CutoffType(1);
  }

  for (std::size_t i = 0; i < n; ++i) {
    // Find the largest positive and negative differences and their positions.
    CutoffType sum = 0;
    CutoffType minval = 0;
    CutoffType maxval = 0;
    IndexType minpos, maxpos;
    for (std::size_t j = 0; j < n; ++j) {
      sum += std::abs(b[j]);
      if (b[j] <= minval) {
        minval = b[j];
        minpos = j;
      }
      if (b[j] >= maxval) {
        maxval = b[j];
        maxpos = j;
      }
    }
    
    if (sum < tol) break;

    // Assign alias and cutoff values
    table[minpos] = std::make_pair(1 + minval, maxpos);
    b[maxpos] += minval;
    b[minpos] = 0;
  }
}

} // end namespace detail

//
// double-based Walker algorithm
//

template<class CutoffType, class IntType, class RealType>
class random_choice_walker;

template<class IntType, class RealType>
class random_choice_walker<double, IntType, RealType> {
public:
  typedef RealType input_type;
  typedef IntType result_type;

  random_choice_walker() {}
  template<class CONT>
  random_choice_walker(const CONT& weights) { detail::fill_ft2009(weights, table_); }

  template<class Engine>
  result_type operator()(Engine& eng) const {
    result_type x = result_type(RealType(size()) * eng());
    return (eng() < cutoff(x)) ? x : alias(x);
  }

  template<class CONT>
  bool check(const CONT& weights, RealType tol = 1.0e-10) const {
    return detail::check_table(weights, table_, tol);
  }

protected:
  IntType size() const { return table_.size(); }
  RealType cutoff(result_type i) const { return table_[i].first; }
  result_type alias(result_type i) const { return table_[i].second; }

private:
  std::vector<std::pair<RealType, result_type> > table_; // first element:  cutoff value
                                                         // second element: alias
};


//
// optimized integer-based version of Walker algorithm
//

template<class IntType, class RealType>
class random_choice_walker<unsigned int, IntType, RealType> {
public:
  typedef IntType input_type;
  typedef IntType result_type;

  random_choice_walker() {}
  template<class CONT>
  random_choice_walker(const CONT& weights) {
    detail::fill_ft2009(weights, table_);
    bits_ = 31 - int(std::log(table_.size() - 0.5) / std::log(2.0));
  }

  template<class Engine>
  result_type operator()(Engine& eng) const {
    result_type x = eng() >> bits_;
    return (eng() < cutoff(x)) ? x : alias(x);
  }

  template<class CONT>
  bool check(const CONT& weights, RealType tol = 1.0e-10) const {
    return detail::check_table(weights, table_, tol);
  }

protected:
  IntType cutoff(IntType i) const { return table_[i].first; }
  IntType alias(IntType i) const { return table_[i].second; }

private:
  IntType bits_; // number of bits to be disposed
  std::vector<std::pair<IntType, IntType> > table_;
};


//
// random_choice_bsearch (O(log N) algorithm using binary search algorithm)
//

template<class IntType = unsigned int, class RealType = double>
class random_choice_bsearch {
public:
  typedef RealType input_type;
  typedef IntType result_type;

  random_choice_bsearch() : accum_(0) {}
  template<class CONT>
  random_choice_bsearch(CONT const& weights) { init(weights); }

  template<class CONT>
  void init(CONT const& weights) {
#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
    BOOST_STATIC_ASSERT(std::numeric_limits<IntType>::is_integer);
    BOOST_STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer);
#endif
    if (weights.size() == 0)
      boost::throw_exception(std::invalid_argument("random_choice_bsearch::init"));
    RealType norm = 0;
    BOOST_FOREACH(RealType w, weights) {
      if (w < RealType(0))
        boost::throw_exception(std::invalid_argument("random_choice_bsearch::init"));
      norm += w;
    }
    if (norm <= RealType(0))
      boost::throw_exception(std::invalid_argument("random_choice_bsearch::init"));
    accum_.resize(0);
    double a = 0;
    BOOST_FOREACH(RealType w, weights) {
      a += w / norm;
      accum_.push_back(a);
    }
  }

  template<class Engine>
  result_type operator()(Engine& eng) const {
    double p = eng();
    int first = 0;
    int last = accum_.size();  // pointing to the next of the last element
    int current = first + ((last - first) >> 1);
    if (last - first == 0) {
      return first;
    } else if (last - first == 1) {
      if (p < accum_[first])
        return first;
      else
        return first + 1;
    }
    while (true) {
      if (last - first > 3) {
        if (p < accum_[current]) {
          last = current;
          current = first + ((current - first) >> 1);
        } else {
          first = current;
          current += ((last - current) >> 1);
        }
      } else if (last - first == 3) {
        if (p < accum_[first])
          return first;
        else if  (p < accum_[first + 1])
          return first + 1;
        else if  (p < accum_[first + 2])
          return first + 2;
        else
          return first + 3;
      } else {
        if (last - first == 2) {
          if (p < accum_[first])
            return first;
          else if  (p < accum_[first + 1])
            return first + 1;
          else
            return first + 2;
        }
      }
    }
  }

private:
  std::vector<RealType> accum_;
};


//
// random_choice_lsearch (O(N) algorithm with naive linear search)
//

template<class IntType = unsigned int, class RealType = double>
class random_choice_lsearch {
public:
  typedef RealType input_type;
  typedef IntType result_type;

  random_choice_lsearch() : accum_(0) {}
  template<class CONT>
  random_choice_lsearch(CONT const& weights) { init(weights); }

  template<class CONT>
  void init(CONT const& weights) {
#ifndef BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS
    BOOST_STATIC_ASSERT(std::numeric_limits<IntType>::is_integer);
    BOOST_STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer);
#endif
    if (weights.size() == 0)
      boost::throw_exception(std::invalid_argument("random_choice_lsearch::init"));
    double norm = 0;
    BOOST_FOREACH(RealType w, weights) {
      if (w < RealType(0))
        boost::throw_exception(std::invalid_argument("random_choice_lsearch::init"));
      norm += w;
    }
    if (norm <= RealType(0))
      boost::throw_exception(std::invalid_argument("random_choice_lsearch::init"));
    accum_.resize(0);
    double a = 0;
    BOOST_FOREACH(RealType w, weights) {
      a += w / norm;
      accum_.push_back(a);
    }
  }

  template<class Engine>
  result_type operator()(Engine& eng) const {
    RealType x = eng();
    for (result_type r = 0; r < accum_.size(); ++r) if (accum_[r] > x) return r;
    return result_type(0); // never reached
  }

private:
  std::vector<RealType> accum_;
};

template<typename RNG>
class random_choice : public random_choice_walker<typename RNG::result_type, unsigned int, double> {
private:
  typedef random_choice_walker<typename RNG::result_type, unsigned int, double> base_type;
public:
  random_choice() : base_type() {}
  template<class CONT>
  random_choice(const CONT& weights) : base_type(weights) {}
};

template<>
class random_choice<double> : public random_choice_walker<double, unsigned int, double> {
private:
  typedef random_choice_walker<double, unsigned int, double> base_type;
public:
  random_choice() : base_type() {}
  template<class CONT>
  random_choice(const CONT& weights) : base_type(weights) {}
};

template<>
class random_choice<unsigned int> : public random_choice_walker<unsigned int, unsigned int, double> {
private:
  typedef random_choice_walker<unsigned int, unsigned int, double> base_type;
public:
  random_choice() : base_type() {}
  template<class CONT>
  random_choice(const CONT& weights) : base_type(weights) {}
};

} // end namespace alps

#endif // ALPS_RANDOM_CHOICE_HPP
