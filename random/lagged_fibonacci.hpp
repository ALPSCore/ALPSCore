/* boost random/lagged_fibonacci.hpp header file
 *
 * Copyright Jens Maurer 2000-2001
 * Permission to use, copy, modify, sell, and distribute this software
 * is hereby granted without fee provided that the above copyright notice
 * appears in all copies and that both that copyright notice and this
 * permission notice appear in supporting documentation,
 *
 * Jens Maurer makes no representations about the suitability of this
 * software for any purpose. It is provided "as is" without express or
 * implied warranty.
 *
 * See http://www.boost.org for most recent version including documentation.
 *
 * $Id$
 *
 * Revision history
 *  2001-02-18  moved to individual header files
 */

#ifndef ALPS_FIBONACCIO_HPP 
#define ALPS_FIBONACCIO_HPP

#include <iostream>
#include <algorithm>     // std::max
#include <iterator>
#include <boost/config.hpp>
#include <boost/limits.hpp>
#include <boost/cstdint.hpp>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_01.hpp>

namespace alps {

template<class RealType, int w, unsigned int p, unsigned int q>
class lagged_fibonacci_01
{
public:
  typedef RealType result_type;
  BOOST_STATIC_CONSTANT(bool, has_fixed_range = false);
  BOOST_STATIC_CONSTANT(int, word_size = w);
  BOOST_STATIC_CONSTANT(unsigned int, long_lag = p);
  BOOST_STATIC_CONSTANT(unsigned int, short_lag = q);

  lagged_fibonacci_01() { init_modulus(); seed(); }
  explicit lagged_fibonacci_01(uint32_t value) { init_modulus(); seed(value); }
  template<class Generator>
  explicit lagged_fibonacci_01(Generator & gen) { init_modulus(); seed(gen); }
  template<class It> lagged_fibonacci_01(It& first, It last)
  { init_modulus(); seed(first, last); }
  // compiler-generated copy ctor and assignment operator are fine

private:
  void init_modulus()
  {
#ifndef BOOST_NO_STDC_NAMESPACE
    // allow for Koenig lookup
    using std::pow;
#endif
    _modulus = pow(RealType(2), word_size);
  }

public:
  void seed(uint32_t value = 331u)
  {
    boost::minstd_rand0 intgen(value);
    seed(intgen);
  }

  // For GCC, moving this function out-of-line prevents inlining, which may
  // reduce overall object code size.  However, MSVC does not grok
  // out-of-line template member functions.
  template<class Generator>
  void seed(Generator & gen)
  {
    boost::uniform_01<Generator, RealType> gen01(gen);
    // I could have used std::generate_n, but it takes "gen" by value
    for(unsigned int j = 0; j < long_lag; ++j)
      x[j] = gen01();
    i = long_lag;
  }

  template<class It>
  void seed(It& first, It last)
  {
#ifndef BOOST_NO_STDC_NAMESPACE
    // allow for Koenig lookup
    using std::fmod;
#endif
    unsigned long mask = ~0u;
    for(int k = 0; k < w; ++k)
      mask <<= 1;
    mask = ~mask;                // now lowest w bits set
    unsigned int j;
    for(j = 0; j < long_lag && first != last; ++j, ++first)
      x[j] = fmod((*first & mask) / _modulus, RealType(1));
    i = long_lag;
    if(first == last && j < long_lag)
      throw std::invalid_argument("lagged_fibonacci_01::seed");
  }

  result_type min() const { return result_type(0); }
  result_type max() const { return result_type(1); }

  result_type operator()()
  {
    if(i >= long_lag)
      fill();
    return x[i++];
  }

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const lagged_fibonacci_01&f)
  {
#ifndef BOOST_NO_STDC_NAMESPACE
    // allow for Koenig lookup
    using std::pow;
#endif
    os << f.i << " ";
    std::ios_base::fmtflags oldflags = os.flags(os.dec | os.fixed | os.left); 
    for(unsigned int i = 0; i < long_lag; ++i)
      os << f.x[i] * f._modulus << " ";
    os.flags(oldflags);
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT, Traits>&
  operator>>(std::basic_istream<CharT, Traits>& is, lagged_fibonacci_01& f)
  {
   is >> f.i >> std::ws;
   for(unsigned int i = 0; i < long_lag; ++i) {
      RealType value;
      is >> value >> std::ws;
      if(value>=1.)
        f.x[i] = value / f._modulus;
      else
        f.x[i] = value;
    }
    return is;
  }

  friend bool operator==(const lagged_fibonacci_01& x,
                         const lagged_fibonacci_01& y)
  { return x.i == y.i && std::equal(x.x, x.x+long_lag, y.x); }
  friend bool operator!=(const lagged_fibonacci_01& x,
                         const lagged_fibonacci_01& y)
  { return !(x == y); }
#else
  // Use a member function; Streamable concept not supported.
  bool operator==(const lagged_fibonacci_01& rhs) const
  { return i == rhs.i && std::equal(x, x+long_lag, rhs.x); }
  bool operator!=(const lagged_fibonacci_01& rhs) const
  { return !(*this == rhs); }
#endif

private:
  void fill();
  unsigned int i;
  RealType x[long_lag];
  RealType _modulus;
};

#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
//  A definition is required even for integral static constants
template<class RealType, int w, unsigned int p, unsigned int q>
const bool lagged_fibonacci_01<RealType, w, p, q>::has_fixed_range;
template<class RealType, int w, unsigned int p, unsigned int q>
const unsigned int lagged_fibonacci_01<RealType, w, p, q>::long_lag;
template<class RealType, int w, unsigned int p, unsigned int q>
const unsigned int lagged_fibonacci_01<RealType, w, p, q>::short_lag;

#endif

template<class RealType, int w, unsigned int p, unsigned int q>
void lagged_fibonacci_01<RealType, w, p, q>::fill()
{
  // two loops to avoid costly modulo operations
  {  // extra scope for MSVC brokenness w.r.t. for scope
  for(unsigned int j = 0; j < short_lag; ++j) {
    RealType t = x[j] + x[j+(long_lag-short_lag)];
    if(t >= RealType(1))
      t -= RealType(1);
    x[j] = t;
  }
  }
  for(unsigned int j = short_lag; j < long_lag; ++j) {
    RealType t = x[j] + x[j-short_lag];
    if(t >= RealType(1))
      t -= RealType(1);
    x[j] = t;
  }
  i = 0;
}


typedef lagged_fibonacci_01<double, 48, 607, 273> lagged_fibonacci607;
typedef lagged_fibonacci_01<double, 48, 1279, 418> lagged_fibonacci1279;
typedef lagged_fibonacci_01<double, 48, 2281, 1252> lagged_fibonacci2281;
typedef lagged_fibonacci_01<double, 48, 3217, 576> lagged_fibonacci3217;
typedef lagged_fibonacci_01<double, 48, 4423, 2098> lagged_fibonacci4423;
typedef lagged_fibonacci_01<double, 48, 9689, 5502> lagged_fibonacci9689;
typedef lagged_fibonacci_01<double, 48, 19937, 9842> lagged_fibonacci19937;
typedef lagged_fibonacci_01<double, 48, 23209, 13470> lagged_fibonacci23209;
typedef lagged_fibonacci_01<double, 48, 44497, 21034> lagged_fibonacci44497;


// It is possible to partially specialize uniform_01<> on lagged_fibonacci_01<>
// to help the compiler generate efficient code.  For GCC, this seems useless,
// because GCC optimizes (x-0)/(1-0) to (x-0).  This is good enough for now.

} // namespace alps

#endif // BOOST_RANDOM_LAGGED_FIBONACCI_HPP
