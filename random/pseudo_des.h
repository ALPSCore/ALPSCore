/***************************************************************************
* ALPS++/alps library
*
* alps/pseudo_des.h    DES-like pseudo random number generator based on 
*                      Numerical Recipes in C, 2nd Ed. (Campbridge, 1992),
*                      Sec 7.5.
*
* $Id$
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*
* This program is free software; you can redistribute it and/or
* modify it under the terms of the GNU General Public License
* as published by the Free Software Foundation; either version 2
* of the License, or (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
**************************************************************************/

#ifndef ALPS_RANDOM_PSEUDO_DES_HPP
#define ALPS_RANDOM_PSEUDO_DES_HPP

#include <alps/config.h>
#include <iostream>
#include <algorithm>  // std::swap
#include <boost/config.hpp>
#include <boost/limits.hpp>
#include <boost/integer_traits.hpp>

namespace alps {

class pseudo_des
{
public:
  typedef uint32_t result_type;
  
#ifndef BOOST_NO_INCLASS_MEMBER_INITIALIZATION
  static const bool has_fixed_range = true;
  static const result_type min_value = boost::integer_traits<result_type>::const_min;
  static const result_type max_value = boost::integer_traits<result_type>::const_max;
#else
  BOOST_STATIC_CONSTANT(bool, has_fixed_range = false);
#endif

  BOOST_STATIC_CONSTANT(uint32_t, default_seed = 4357);

  result_type min() const { return std::numeric_limits<result_type>::min(); }
  result_type max() const { return std::numeric_limits<result_type>::max(); }

  pseudo_des() : seed_(default_seed), state_(1) {}
  explicit pseudo_des(uint32_t s) : seed_(s), state_(1) {}

  // compiler-generated copy ctor and assignment operator are fine

  void seed(uint32_t s=default_seed) {
    seed_ = s;
    state_ = 1;
  }

  result_type operator()() {
    return hash(seed_, state_++);
  }

  void operator+=(uint32_t skip) { state_ += skip; }

  std::ostream& write(std::ostream& os) const {
    return os << seed_ << ' ' << state_;
  }
  std::istream& read(std::istream& is) {
    return is >> seed_ >> state_;
  }
  bool operator==(const pseudo_des& rhs) const {
    return seed_ == rhs.seed_ && state_ == rhs.state_;
  }

protected:
  uint32_t low_bits(uint32_t d) const {
    return d & 0xffff;
  }
  uint32_t high_bits(uint32_t d) const {
    return d >> 16;
  }
  uint32_t swap_bits(uint32_t d) const {
    return high_bits(d) | (low_bits(d) << 16);
  }

  uint32_t hash(uint32_t w0, uint32_t w1) const {
    static const int num_iter = 4;
    static const uint32_t g0[num_iter] = {
      0xbaa96887L, 0x1e17d32cL, 0x03bcdc3cL, 0x0f33d1b2L
    };
    static const uint32_t g1[num_iter] = {
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

private:
  uint32_t seed_;
  uint32_t state_;
};

} // end namespace alps

#endif // RANDOM_PSEUDO_DES_HPP
