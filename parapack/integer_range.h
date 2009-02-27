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

#ifndef PARAPACK_INTEGER_RANGE_H
#define PARAPACK_INTEGER_RANGE_H

#include <alps/expression.h>
#include <alps/osiris.h>
#include <boost/config.hpp>
#include <boost/call_traits.hpp>
#include <boost/classic_spirit.hpp>
#include <boost/throw_exception.hpp>
#include <iosfwd>
#include <limits>
#include <stdexcept>
#include <string>

namespace alps {

//
// class tempalte integer_range
//

template<class T>
class integer_range {
public:
  typedef T value_type;
  typedef typename boost::call_traits<value_type>::param_type param_type;

  integer_range() : mi_(1), ma_(0) {}
  explicit integer_range(param_type v) : mi_(v), ma_(v) {}
  explicit integer_range(param_type vmin, param_type vmax) : mi_(vmin), ma_(vmax) {
    if (mi_ > ma_) clear();
  }
  integer_range(integer_range const& r) : mi_(r.mi_), ma_(r.ma_) {}
  integer_range(std::string const& str) :
    mi_(std::numeric_limits<value_type>::min BOOST_PREVENT_MACRO_SUBSTITUTION ()), ma_(std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) {
    init(str, Parameters());
  }
  integer_range(std::string const& str, Parameters const& p) :
    mi_(std::numeric_limits<value_type>::min BOOST_PREVENT_MACRO_SUBSTITUTION ()), ma_(std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ()) {
    init(str, p);
  }
  integer_range(std::string const& str, Parameters const& p, param_type def_mi,
    param_type def_ma) : mi_(def_mi), ma_(def_ma) {
    init(str, p);
  }

  integer_range& operator=(param_type v) {
    mi_ = ma_ = v;
    return *this;
  }

  bool operator==(integer_range const& rhs) const { return mi_ == rhs.mi_ && ma_ == rhs.ma_; }
  bool operator!=(integer_range const& rhs) const { return !operator==(rhs); }

  template<typename U>
  integer_range& operator*=(U x) {
    if (x > U(0)) {
      if (x > U(1) && mi_ > static_cast<value_type>(std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION () / x))
        mi_ = std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
      else
        mi_ = static_cast<value_type>(x * mi_);
      if (x > U(1) && ma_ > static_cast<value_type>(std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION () / x))
        ma_ = std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
      else
        ma_ = static_cast<value_type>(x * ma_);
    } else {
      boost::throw_exception(std::runtime_error("integer_range: multiplier should be positive"));
    }
    return *this;
  }

  void clear() {
    mi_ = 1;
    ma_ = 0;
  }
  void include(param_type v) {
    if (valid()) {
      mi_ = std::min(mi_, v);
      ma_ = std::max(ma_, v);
    } else {
      mi_ = v;
      ma_ = v;
    }
  }
  void include(integer_range const& r) {
    if (r.valid()) {
      if (valid()) {
        mi_ = std::min(mi_, r.min BOOST_PREVENT_MACRO_SUBSTITUTION ());
        ma_ = std::max(ma_, r.max BOOST_PREVENT_MACRO_SUBSTITUTION ());
      } else {
        mi_ = r.min BOOST_PREVENT_MACRO_SUBSTITUTION ();
        ma_ = r.max BOOST_PREVENT_MACRO_SUBSTITUTION ();
      }
    }
  }

  value_type min BOOST_PREVENT_MACRO_SUBSTITUTION () const { return mi_; }
  value_type max BOOST_PREVENT_MACRO_SUBSTITUTION () const { return ma_; }
  value_type size() const { return 1 + ma_ - mi_; }
  bool empty() const { return size() == 0; }
  bool valid() const { return size() != 0; }
  bool is_included(param_type v) const { return (v >= min BOOST_PREVENT_MACRO_SUBSTITUTION ()) && (v <= max BOOST_PREVENT_MACRO_SUBSTITUTION ()); }

  integer_range overlap(integer_range const& r) const {
    return (empty() || r.empty()) ? integer_range()
      : integer_range(std::max BOOST_PREVENT_MACRO_SUBSTITUTION (min BOOST_PREVENT_MACRO_SUBSTITUTION (), r.min BOOST_PREVENT_MACRO_SUBSTITUTION ()), std::min BOOST_PREVENT_MACRO_SUBSTITUTION (max BOOST_PREVENT_MACRO_SUBSTITUTION (), r.max BOOST_PREVENT_MACRO_SUBSTITUTION ()));
  }

  void save(ODump& dp) const { dp << mi_ << ma_; }
  void load(IDump& dp) { dp >> mi_ >> ma_; }

protected:
  void init(std::string const& str, Parameters const& p) {
    using namespace boost::spirit;
    std::string mi_str, ma_str;
    if (!parse(
      str.c_str(),
          ( ch_p('[')
            >> (*(anychar_p-'['-':'-']'))[assign_a(mi_str)]
            >> ':'
            >> (*(anychar_p-'['-':'-']'))[assign_a(ma_str)]
            >> ']' )
        | ( ch_p('[')
            >> (+(anychar_p-'['-':'-']'))[assign_a(mi_str)][assign_a(ma_str)]
            >> ']' )
        | ( ch_p('[') >> ']')
        | (+(anychar_p-'['-':'-']'))[assign_a(mi_str)][assign_a(ma_str)]
        >> end_p,
      space_p).full)
      boost::throw_exception(std::runtime_error("integer_range: parse error: " + str));
    if (mi_str.empty() && ma_str.empty()) {
      mi_str = "1";
      ma_str = "0";
    }
    if (mi_str.empty()) {
      mi_ = std::numeric_limits<value_type>::min BOOST_PREVENT_MACRO_SUBSTITUTION ();
    } else {
      double v = alps::evaluate(mi_str, p);
      if (v < std::numeric_limits<value_type>::min BOOST_PREVENT_MACRO_SUBSTITUTION () ||
          v > std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
        boost::throw_exception(std::runtime_error("integer_range: range error"));
      mi_ = static_cast<value_type>(v);
      // Note: boost::numeric_cast<value_type>(v) does not work for intel C++ for x86_64
    }
    if (ma_str.empty()) {
      ma_ = std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
    } else {
      double v = alps::evaluate(ma_str, p);
      if (v < std::numeric_limits<value_type>::min BOOST_PREVENT_MACRO_SUBSTITUTION () ||
          v > std::numeric_limits<value_type>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
        boost::throw_exception(std::runtime_error("integer_range: range error"));
      ma_ = static_cast<value_type>(v);
      // Note: boost::numeric_cast<value_type>(v) does not work for intel C++ for x86_64
    }
    if (mi_ > ma_) clear();
  }

private:
  value_type mi_, ma_;
};

template<typename T>
integer_range<T> overlap(integer_range<T> const& r0, integer_range<T> const& r1) {
  return r0.overlap(r1);
}

template<typename T>
integer_range<T> unify(integer_range<T> const& r0, integer_range<T> const& r1) {
  if (r0.empty()) {
    return r1;
  } else if (r1.empty()) {
    return r0;
  } else {
    if (overlap(r0, r1).size() == 0)
      boost::throw_exception(std::range_error("no overlap"));
    integer_range<T> res = r0;
    res.include(r1);
    return res;
  }
}

} // end namespace alps

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template<typename T, typename U>
alps::integer_range<T> operator*(alps::integer_range<T> const& t, U x) {
  alps::integer_range<T> result = t;
  result *= x;
  return result;
}

template<typename T, typename U>
alps::integer_range<T> operator*(U x, alps::integer_range<T> const& t) {
  return t * x;
}

template<class T>
std::ostream& operator<<(std::ostream& os, integer_range<T> const& ir) {
  if (ir.valid())
    os << '[' << ir.min BOOST_PREVENT_MACRO_SUBSTITUTION () << ':' << ir.max BOOST_PREVENT_MACRO_SUBSTITUTION () << ']';
  else
    os << "[]";
  return os;
}

template<class T>
alps::ODump& operator<<(alps::ODump& dp, alps::integer_range<T> const& ir) {
  ir.save(dp);
  return dp;
}

template<class T>
alps::IDump& operator>>(alps::IDump& dp, alps::integer_range<T>& ir) {
  ir.load(dp);
  return dp;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif // PARAPACK_INTEGER_RANGE_H
