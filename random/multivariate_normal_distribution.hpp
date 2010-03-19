/* boost random/normal_distribution.hpp header file
 *
 * Copyright Matthias Troyer and Wesley P. Petersen 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 */

#ifndef ALPS_RANDOM_MULTIVARIATE_NORMAL_DISTRIBUTION_HPP
#define ALPS_RANDOM_MULTIVARIATE_NORMAL_DISTRIBUTION_HPP

#include <cmath>
#include <cassert>
#include <iostream>
#include <boost/limits.hpp>
#include <boost/assert.hpp>
#include <boost/static_assert.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/random/variate_generator.hpp>
#include <boost/random/normal_distribution.hpp>

/// @file This header provides a multi-variate normal distribution of correlated normally distributed random numbers.

namespace alps {

/// @brief a distribution creating correlated multi-variate normally distributed numbers
///
/// the multivariate normal distribution creates correlated random numbers with a specified
/// mean vector and covariance matrix. Instead of using a vector as the @c result_type, the
/// @c result_type is a scalar real number, and the @c operator() should be called once for
/// each element of the vector.

template<class RealType = double>
class multivariate_normal_distribution
{
public:
  /// the type of random numbers produced  
  typedef RealType result_type;

  /// the input type
  typedef RealType input_type;
  
  
  /// the type of the vector of mean values
  typedef boost::numeric::ublas::vector<RealType> vector_type;
  
  /// the type of the matrix storing the Cholesky decomposition of the covariance matrix 
  typedef boost::numeric::ublas::matrix<RealType> matrix_type;

#if !defined(BOOST_NO_LIMITS_COMPILE_TIME_CONSTANTS) && !(defined(BOOST_MSVC) && BOOST_MSVC <= 1300)
/// INTERNAL ONLY
    BOOST_STATIC_ASSERT(!std::numeric_limits<RealType>::is_integer);
#endif

  /// @brief the constructor of the multi-variate normal distribution
  /// @param mean the vector of mean values
  /// @param cholesky the Cholesky decomposition of the covariance matrix
  ///
  /// The Cholesky decomposition has to be a square matrix and its dimension
  /// has to be the same as that of the mean vector. Instead of a Cholesky
  /// decomposition any other square root of the covariance matrix could be passed.
  ///
  /// Requires:  mean.size() == cholesky.size1() and cholesky.size1() == cholesky.size2()

  
  multivariate_normal_distribution(const matrix_type& cholesky, const vector_type& mean)
    : mean_(mean)
    , cholesky_(cholesky)
    , buffer_(mean_.size())
    , ptr_(buffer_.end())
  {
    BOOST_ASSERT(mean_.size()==cholesky_.size1() && mean_.size()==cholesky_.size2());
  }

  /// @brief the constructor of the multi-variate normal distribution with zero mean
  /// @param cholesky the Cholesky decomposition of the covariance matrix
  ///
  /// Instead of a Cholesky decomposition any other square root 
  /// of the covariance matrix could be passed.
  ///
  /// Requires: cholesky.size1() == cholesky.size2()

  
  explicit multivariate_normal_distribution(const matrix_type& cholesky)
    : mean_(cholesky.size1())
    , cholesky_(cholesky)
    , buffer_(mean_.size())
    , ptr_(buffer_.end())
  {
    mean_.clear();
  }

/// INTERNAL ONLY
  multivariate_normal_distribution(const multivariate_normal_distribution& other)
    : mean_(other.mean_)
    , cholesky_(other.cholesky_)
    , buffer_(mean_.size())
    , ptr_(buffer_.end())
  {
  }

/// INTERNAL ONLY
  multivariate_normal_distribution& operator=(const multivariate_normal_distribution& other)
  {
    mean_=other.mean_;
    cholesky_=other.cholesky_;
    buffer_=other.buffer_;
    ptr_ = buffer_.begin() + (other.ptr_-other.buffer_.begin());
    return *this;
  }

  /// @return the vector of mean values
  vector_type const& mean() const { return mean_; }

  /// @return the Cholesky decomposition of the covariance marix
  matrix_type const& cholesky() const { return cholesky_; }

  ///  @brief purges the cache. 
  ///
  /// Purges the cache. The next call to oeprator() will create a new vector.
  
  void reset() { ptr_ = buffer_.end(); }


  /// @return On the first call, it creates a vector of n multivariate normally 
  /// distributed random numbers, caches it, and returns the first value. The next n-1 calls 
  /// return the next elements of the vector. The n+1-st call again creates a new vector
  /// of multivariate normally distributed random numbers.


  template<class Engine>
  result_type operator()(Engine& eng)
  {
    if(ptr_ == buffer_.end()) {
      boost::variate_generator<Engine&,boost::normal_distribution<RealType> > 
        gen(eng,boost::normal_distribution<RealType>());
      std::generate(buffer_.begin(),buffer_.end(),gen);
      buffer_=boost::numeric::ublas::prod(cholesky_,buffer_)+mean_;
      ptr_ = buffer_.begin();
    }
    return *ptr_++;
  }

#if !defined(BOOST_NO_OPERATORS_IN_NAMESPACE) && !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const multivariate_normal_distribution& mnd)
  {
    std::size_t pos = mnd.ptr_-mnd.buffer_.begin();
    os << mnd.mean_ << mnd.cholesky_ << mnd.buffer_ << pos;
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is, multivariate_normal_distribution& mnd)
  {
    std::size_t pos;
    is >> std::ws >> mnd.mean_ >> std::ws >> mnd.cholesky_
       >> std::ws >> mnd.buffer_ >> std::ws >> pos;
    mnd.ptr_ = mnd.buffer_.begin() + pos;
    return is;
  }
#endif
private:
  vector_type mean_;
  matrix_type cholesky_;
  vector_type buffer_;
  typename vector_type::const_iterator ptr_;
};

} // namespace alps

#endif // ALPS_RANDOM_MULTIVARIATE_NORMAL_DISTRIBUTION_HPP
