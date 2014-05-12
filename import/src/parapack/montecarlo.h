/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1997-2009 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef PARAPACK_MONTECARLO_H
#define PARAPACK_MONTECARLO_H

#include "integer_range.h"

#include <alps/expression.h>
#include <alps/osiris.h>
#include <boost/throw_exception.hpp>
#include <stdexcept>

namespace alps {

class mc_steps {
public:
  typedef integer_range<unsigned int> range_type;

  mc_steps() : mcs_(0), sweep_("[0:]"), therm_(0) {}
  mc_steps(alps::Parameters const& p) : mcs_(0),
    sweep_(p.value_or_default("SWEEPS", "[65536:]"), p),
    therm_(p.defined("THERMALIZATION") ? static_cast<int>(alps::evaluate("THERMALIZATION", p)) :
      (sweep_.min BOOST_PREVENT_MACRO_SUBSTITUTION () >> 3)) {
  }

  void set_thermalization(int c) { therm_ = c; }
  void set_sweeps(unsigned int c) { sweep_ = range_type(c); }
  void set_sweeps(range_type const& c) { sweep_ = c; }

  mc_steps& operator++() { ++mcs_; return *this; }
  mc_steps operator++(int) { mc_steps tmp = *this; this->operator++(); return tmp; }

  unsigned int operator()() const { return mcs_; }
  bool can_work() const {
    return mcs_ < therm_ || mcs_ - therm_ < sweep_.max BOOST_PREVENT_MACRO_SUBSTITUTION ();
  }
  bool is_thermalized() const { return mcs_ >= therm_; }
  double progress() const {
    return static_cast<double>(mcs_) / (therm_ + sweep_.min BOOST_PREVENT_MACRO_SUBSTITUTION());
  }

  int thermalization() const { return therm_; }
  range_type sweeps() const { return sweep_; }

  void save(alps::ODump& dp) const { dp << mcs_ << sweep_ << therm_; }
  void load(alps::IDump& dp) { dp >> mcs_ >> sweep_ >> therm_; }

private:
  unsigned int mcs_;
  range_type sweep_;
  unsigned int therm_;
};

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::ODump& operator<<(alps::ODump& dp, alps::mc_steps const& mcs) {
  mcs.save(dp);
  return dp;
}

inline alps::IDump& operator>>(alps::IDump& dp, alps::mc_steps& mcs) {
  mcs.load(dp);
  return dp;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

namespace alps {

struct hybrid_weight_parameter {
public:
  explicit hybrid_weight_parameter(int x = 0) : energy_(x), order_(x) {
    if (x != 0) boost::throw_exception(std::invalid_argument("hybrid_weight_parameter"));
  }
  hybrid_weight_parameter(double total_energy, double expansion_order) :
    energy_(total_energy), order_(expansion_order) {}
  double log_weight(double beta) const { return -energy_ * beta + order_ * std::log(beta); }
  hybrid_weight_parameter& operator+=(hybrid_weight_parameter const& wp) {
    energy_ += wp.energy_;
    order_ += wp.order_;
    return *this;
  }
  hybrid_weight_parameter operator+(hybrid_weight_parameter const& wp) const {
    return hybrid_weight_parameter(energy_ + wp.energy_, order_ + wp.order_);
  }
  hybrid_weight_parameter operator*(double x) const {
    return hybrid_weight_parameter(energy_*x, order_*x);
  }
  hybrid_weight_parameter operator/(double x) const { return (*this) * (1/x); }

  void save(alps::ODump& dp) const { dp << energy_ << order_; }
  void load(alps::IDump& dp) { dp >> energy_ >> order_; }
  template<typename Archive>
  void serialize(Archive& ar, const unsigned int) { ar & energy_ & order_; }

private:
  double energy_;
  double order_;
};

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

inline alps::ODump& operator<<(alps::ODump& dp, alps::hybrid_weight_parameter const& wp) {
  wp.save(dp);
  return dp;
}

inline alps::IDump& operator>>(alps::IDump& dp, alps::hybrid_weight_parameter& wp) {
  wp.load(dp);
  return dp;
}

inline alps::hybrid_weight_parameter operator*(double x, alps::hybrid_weight_parameter const& wp) {
  return wp * x;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // PARAPACK_MONTECARLO_H
