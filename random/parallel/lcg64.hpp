/* boost random/parallel/lcg64.hpp header file
 *
 * Copyright Matthias Troyer 2006
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE_1_0.txt or copy at
 * http://www.boost.org/LICENSE_1_0.txt)
 *
 */

/// \file parallel/lcg64.hpp
/// 
/// contains a parallel 64-bit generator template and three good choices
/// of parameters.

#ifndef ALPS_RANDOM_PARALLEL_LCG64_HPP
#define ALPS_RANDOM_PARALLEL_LCG64_HPP

#include <iostream>
#include <cassert>
#include <stdexcept>
#include <boost/config.hpp>
#include <boost/limits.hpp>
#include <boost/static_assert.hpp>
#include <boost/detail/workaround.hpp>
#include <boost/parameter/macros.hpp>
#include <boost/preprocessor/cat.hpp>
#include <boost/assert.hpp>
#include <boost/integer_traits.hpp>
#include <alps/random/parallel/keyword.hpp>
#include <alps/random/parallel/detail/get_prime.hpp>
#include <alps/random/parallel/detail/seed_macros.hpp>

#if !defined(BOOST_NO_INT64_T) && !defined(BOOST_NO_INTEGRAL_INT64_T)

namespace alps {
namespace random {
namespace parallel {

/// @brief a 64-bit linear congruential generator
///
/// A 64-bit parallel linear congruential generator, 
/// following the implementation of the SPRNG library

template<uint64_t a, uint64_t val>
class lcg64
{
public:
  /// The result type is a 64-bit unsigned integer
  typedef uint64_t result_type;
#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
  /// @brief This generator has a fixed range
  static const bool has_fixed_range = true;
  /// @brief The minimum vaue is 0
  static const result_type min_value = 0;
  /// @brief The maximum value is the largest unsigned 64 bit integer
  static const result_type max_value =boost::integer_traits<uint64_t>::const_max;
  /// @brief The maximum number of streams is 146138719, the number of primes among all 64-bit unsigned integer values
  static const result_type max_streams = 146138719;
#else
  BOOST_STATIC_CONSTANT(bool, has_fixed_range = false);
  BOOST_STATIC_CONSTANT(result_type, max_streams = 146138719);
#endif

/*
// forward seeding functions with iterator buffers to named versions
#define ALPS_LCG64_SEED_IT(z, n, unused)                                          \
  template <class It BOOST_PP_COMMA_IF(n) BOOST_PP_ENUM_PARAMS(n,class T)>        \
  void seed(It& first, It const& last BOOST_PP_COMMA_IF(n)                        \
               BOOST_PP_ENUM_BINARY_PARAMS(n,T,const& x))                         \
  {                                                                               \
    if(first == last)                                                             \
      throw std::invalid_argument("boost::sprng::parallel::lcg64::seed");         \
    seed(global_seed=*first++ BOOST_PP_COMMA_IF(n) BOOST_PP_ENUM_PARAMS(n,x) );   \
  }
   
BOOST_PP_REPEAT_FROM_TO(0, ALPS_RANDOM_MAXARITY, BOOST_LCG64_SEED_IT,~)

#undef ALPS_LCG64_SEED_IT
*/

#ifdef ALPS_DOXYGEN

  /// @brief the constructors
  ///
  /// All standard and named parameter constructors of random number generator and parallel random number generators are provided
  lcg64(...);

  /// @brief the seed fuctions
  ///
  /// All standard and named parameter seed functions of random number generator and parallel random number generators are provided
  void seed(...);

  /// @returns the minimum value 0
  result_type min  () const;
  
  /// @returns the maximum value, the largest unsigned 64-bit integer
  result_type max  () const;
  

#else
  // forwarding named seeding functions
  ALPS_RANDOM_PARALLEL_SEED(lcg64)
  {
    unsigned int stream = p[stream_number|0u];
    unsigned int s=p[global_seed|0u];
    BOOST_ASSERT(stream < p[total_streams|1u]);
    BOOST_ASSERT(p[total_streams|1u] < max_streams);
    // seed the generator
    c = detail::get_prime_64(stream);
    _x =  (uint64_t(0x2bc6ffffU)<<32 | 0x8cfe166dU)^((uint64_t(s)<<33)|stream);
    // and advance it a bit
    for(uint64_t i=0; i<127*stream; i++)
      operator()();
  }

  ALPS_RANDOM_PARALLEL_ITERATOR_SEED_DEFAULT()

  result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return 0; }
  result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return std::numeric_limits<result_type>::max(); }
#endif

  /// @returns the next random number
  result_type operator()()
  {
    _x = a * _x + c;
    return _x;
  }

  /// @brief the validation function
  ///
  /// The validation function checks whether the passed value is the 10'000-th integer generated from a default-seeded generator 
  static bool validation(uint64_t x) { return x==val; }

#ifdef BOOST_NO_OPERATORS_IN_NAMESPACE
    
  // Use a member function; Streamable concept not supported.
  bool operator==(const lcg64& rhs) const
  { return _x == rhs._x && c==rhs.c; }
  bool operator!=(const lcg64& rhs) const
  { return !(*this == rhs); }

#else 
  friend bool operator==(const lcg64& x,
                         const lcg64& y)
  { return x._x == y._x && x.c == y.c; }
  friend bool operator!=(const lcg64& x,
                         const lcg64& y)
  { return !(x == y && x.c == y.c); }
    
#if !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS) && !BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x551))
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os,
             const lcg64& lcg)
  {
    return os << lcg._x << " " << lcg.c;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is,
             lcg64& lcg)
  {
    return is >> lcg._x >> lcg.c;
  }
 
private:
#endif
#endif
    
  uint64_t _x;
  uint64_t c;
};

#if defined(BOOST_NO_OPERATORS_IN_NAMESPACE) || defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS) || BOOST_WORKAROUND(__BORLANDC__, BOOST_TESTED_AT(0x551))
template<class CharT, class Traits, uint64_t a, unit64_t val>
std::basic_ostream<CharT,Traits>&
operator<<(std::basic_ostream<CharT,Traits>& os,
           const lcg64<a,val>& lcg)
{
    return os << lcg._x << " " << lcg.c;
}

template<class CharT, class Traits, uint64_t a, unit64_t val>
std::basic_istream<CharT,Traits>&
operator>>(std::basic_istream<CharT,Traits>& is,
           lcg64<a,val>& lcg)
{
    return is >> lcg._x >> lcg.c;
}
#endif

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
//  A definition is required even for integral static constants
template<uint64_t a, uint64_t val>
const bool lcg64<a,val>::has_fixed_range;
template<uint64_t a, uint64_t val>
const uint64_t lcg64<a,val>::min_value;
template<uint64_t a, uint64_t val>
const uint64_t lcg64<a,val>::max_value;
template<uint64_t a, uint64_t val>
const uint64_t lcg64<a,val>::max_streams;
#endif

} } // namespace random::parallel

// the three tested versions, validation still missing

/// Instantiation of lcg64 with a  good choice of multiplier
typedef random::parallel::lcg64<uint64_t(0x87b0b0fdU)|uint64_t(0x27bb2ee6U)<<32,
                      uint64_t( 481823773Ul)+(uint64_t(3380683238Ul)<<32)> lcg64a;

/// Instantiation of lcg64 with a  good choice of multiplier
typedef random::parallel::lcg64<uint64_t(0xe78b6955U)|uint64_t(0x2c6fe96eU)<<32,
                      uint64_t(3274024413Ul)+(uint64_t(3475904802Ul)<<32)> lcg64b;

/// Instantiation of lcg64 with a  good choice of multiplier
typedef random::parallel::lcg64<uint64_t(0x31a53f85U)|uint64_t(0x369dea0fU)<<32,
                      uint64_t( 950651229Ul)+(uint64_t(3996309981Ul)<<32)> lcg64c;

} // namespace alps

#endif // !BOOST_NO_INT64_T && !BOOST_NO_INTEGRAL_INT64_T
#endif // ALPS_RANDOM_PARALLEL_LCG64_HPP
