/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2009 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>
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

/// \file pseudo_des.h
/// \brief a random number generator using the pseudo-DES algorithm

#ifndef ALPS_RANDOM_PSEUDO_DES_HPP
#define ALPS_RANDOM_PSEUDO_DES_HPP

#include <alps/config.h>
#include <iostream>
#include <algorithm>  // std::swap
#include <boost/config.hpp>
#include <boost/limits.hpp>
#include <boost/integer_traits.hpp>
#include <boost/detail/workaround.hpp>

namespace alps {

/// \brief A random number generator using the pseudo-DES algorithm
/// 
/// The random number generator follows the BOOST (standard C++)
/// specifications

class pseudo_des
{
public:
  /// type of the random numbers
  typedef uint32_t result_type;
  
#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
  /// the range is fixed
  static const bool has_fixed_range = true;
  /// minimum value is 0
  static const result_type min_value = boost::integer_traits<result_type>::const_min;
  /// maximum value is 2^32-1
  static const result_type max_value = boost::integer_traits<result_type>::const_max;
#else
  BOOST_STATIC_CONSTANT(bool, has_fixed_range = false);
#endif

  /// the default seed is 4357
  BOOST_STATIC_CONSTANT(uint32_t, default_seed = 4357);

  /// minim value is 0
  result_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return min_value; }
  /// maximum value is 2^32-1
  result_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return max_value; }
  /// the default constructor
  pseudo_des() : seed_(default_seed), state_(1) {}
  /// construct with specified seed
  explicit pseudo_des(uint32_t s) : seed_(s), state_(1) {}

  // compiler-generated copy ctor and assignment operator are fine
  
  /// seed the generator
  void seed(uint32_t s=default_seed) {
    seed_ = s;
    state_ = 1;
  }

  /// get the next value
  result_type operator()() { return hash(seed_, state_++); }

  /// skip forward by \a skip numbers
  void operator+=(uint32_t skip) { state_ += skip; }

  /// write the state to a std::ostream
  std::ostream& write(std::ostream& os) const {
    return os << seed_ << ' ' << state_;
  }
  
  /// read the state from a std::istream
  std::istream& read(std::istream& is) {
    return is >> seed_ >> state_;
  }
  
  /// check whether the initial seed and current state of two generators is the same
  bool operator==(const pseudo_des& rhs) const {
    return seed_ == rhs.seed_ && state_ == rhs.state_;
  }

  static uint32_t hash(uint32_t w0, uint32_t w1) {
    const int num_iter = 4;
    const uint32_t g0[num_iter] = {
      0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L
    };
    const uint32_t g1[num_iter] = {
      0x4b0f3b58L, 0xe874f0c3L, 0x6955c5a6L, 0x55a7ca46L
    };
    for (int i = 0; i < num_iter; ++i) {
      std::swap(w0, w1);
      uint32_t mask = w0 ^ g0[i];
      uint32_t lbm = low_bits(mask);
      uint32_t hbm = high_bits(mask);
      w1 ^= (swap_bits(lbm * lbm + ~(hbm * hbm)) ^ g1[i]) + lbm * hbm;
    }
    return w1;
  }

protected:
  static uint32_t low_bits(uint32_t d) { return d & 0xffff; }
  static uint32_t high_bits(uint32_t d) { return d >> 16; }
  static uint32_t swap_bits(uint32_t d)
  { return high_bits(d) | (low_bits(d) << 16); }

private:
  uint32_t seed_;
  uint32_t state_;
};

} // end namespace alps

#endif // RANDOM_PSEUDO_DES_HPP
