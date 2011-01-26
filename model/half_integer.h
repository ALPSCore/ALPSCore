/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2003-2010 by Matthias Troyer <troyer@comp-phys.org>,
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

#ifndef ALPS_MODEL_HALF_INTEGER_H
#define ALPS_MODEL_HALF_INTEGER_H

#include <boost/config.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/not.hpp>
#include <boost/throw_exception.hpp>
#include <boost/type.hpp>
#include <boost/type_traits/is_float.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/type_traits/is_same.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/cast.hpp>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace alps {

template <class I>
class half_integer {
public:
  typedef I integer_type;

  half_integer() : val_(0) {}

  template<class J>
  half_integer(const half_integer<J>& x) : val_(x.get_twice()) {}

#ifndef BOOST_NO_SFINAE
  template <typename J>
  half_integer(J x, typename
    boost::enable_if<boost::is_integral<J> >::type* = 0)
    : val_(2*boost::numeric_cast<I>(x)) {}
  template <typename J>
  half_integer(J x, typename boost::enable_if<
                  boost::mpl::and_<
                     boost::is_float<J>,
                     boost::mpl::not_<boost::is_same<J,double> > 
                  > >::type* = 0)
     : val_(integer_type(std::floor(2*x+0.5))) {}
#else
  half_integer(int x) : val_(2*boost::numeric_cast<I>(x)) {}
#endif

  // catch-all for implicit conversions
  half_integer(double x) : val_(integer_type(std::floor(2*x+0.5))) {}

  // compiler-generated assignment and dtor are OK

  double to_double() const { return 0.5*val_; }
  integer_type to_integer() const 
  { 
    if (get_twice()%2!=0) 
      boost::throw_exception(std::runtime_error("Cannot convert odd half-integer to integer"));
    return get_twice()/2;
  }

  void set_half(integer_type x) { val_=x; }
  integer_type get_twice() const { return val_; }
  bool is_odd() const { return (val_ % 2) != 0;}
  bool is_even() const { return (val_ % 2) == 0;}

  template <class J> bool operator==(const half_integer<J>& rhs) const
  { return val_ == rhs.get_twice(); }
  template <class J> bool operator!=(const half_integer<J>& rhs) const
  { return val_ != rhs.get_twice(); }
  template <class J> bool operator<(const half_integer<J>& rhs) const
  { return val_ < rhs.get_twice(); }
  template <class J> bool operator>(const half_integer<J>& rhs) const
  { return val_ > rhs.get_twice(); }
  template <class J> bool operator<=(const half_integer<J>& rhs) const
  { return val_ <= rhs.get_twice(); }
  template <class J> bool operator>=(const half_integer<J>& rhs) const
  { return val_ >= rhs.get_twice(); }

  bool operator==(double rhs) const { return *this == half_integer(rhs); }
  bool operator!=(double rhs) const { return *this != half_integer(rhs); }
  bool operator<(double rhs) const { return *this < half_integer(rhs); }
  bool operator>(double rhs) const { return *this > half_integer(rhs); }
  bool operator<=(double rhs) const { return *this <= half_integer(rhs); }
  bool operator>=(double rhs) const { return *this >= half_integer(rhs); }

  half_integer operator-() const { half_integer tmp(*this); tmp.val_=-val_; return tmp; }

  half_integer& operator++() { val_ += 2; return *this; }
  half_integer& operator--() { val_ -= 2; return *this; }
  half_integer operator++(int)
  { half_integer tmp(*this); ++(*this); return tmp; }
  half_integer operator--(int)
  { half_integer tmp(*this); --(*this); return tmp; }

  template <class J>
  half_integer& operator+=(const half_integer<J>& x)
  { val_ += x.get_twice(); return *this; }
  template <class J>
  half_integer& operator-=(const half_integer<J>& x)
  { val_ -= x.get_twice(); return *this; }
  half_integer& operator+=(double x)
  { *this += half_integer(x); return *this; }
  half_integer& operator-=(double x)
  { *this -= half_integer(x); return *this; }

  template <class J>
  half_integer operator+(const half_integer<J>& x) const
  { half_integer res(*this); return res += x; }
  template <class J>
  half_integer operator-(const half_integer<J>& x) const
  { half_integer res(*this); return res -= x; }
  half_integer operator+(double x) const
  { half_integer res(*this); return res += x; }
  half_integer operator-(double x) const
  { half_integer res(*this); return res -= x; }

  integer_type distance(const half_integer& x) const
  {
    if ((*this==max BOOST_PREVENT_MACRO_SUBSTITUTION ()) !=
        (x==max BOOST_PREVENT_MACRO_SUBSTITUTION ()))
      return std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
    if (std::numeric_limits<I>::is_signed &&
        (*this==min BOOST_PREVENT_MACRO_SUBSTITUTION ()) !=
        (x==min BOOST_PREVENT_MACRO_SUBSTITUTION ()))
      return std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ();
    assert(static_cast<integer_type>(std::abs(val_)) % 2 == static_cast<integer_type>(std::abs(x.get_twice())) % 2);
    return (val_-x.get_twice())/2;
  }
  static half_integer max BOOST_PREVENT_MACRO_SUBSTITUTION ()
  {
    return half_integer(std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION (),
                        to_distinguish());
  }
  static half_integer min BOOST_PREVENT_MACRO_SUBSTITUTION ()
  {
    return std::numeric_limits<I>::is_signed ?
      -half_integer(std::numeric_limits<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION (),
                    to_distinguish()) :
      half_integer(std::numeric_limits<I>::min BOOST_PREVENT_MACRO_SUBSTITUTION (),
                   to_distinguish());
  }
  
  half_integer abs() const
  { return half_integer(std::abs(val_), to_distinguish()); }

private:
  struct to_distinguish {};
  half_integer(integer_type i, const to_distinguish&)
    : val_(i) {}
  integer_type val_;
};

template <class I>
inline double to_double(const half_integer<I>& x) 
{ 
  return x.to_double();
}

template <class I>
inline I to_integer(const half_integer<I>& x) 
{ 
  return x.to_integer();
}

template <class T>
inline bool is_odd(T x)
{
  return (x%2);
}

template <class I>
inline bool is_odd(const half_integer<I>& x)
{
  return (x.abs().get_twice()%4 == 2);
}

template <class I>
inline half_integer<I> abs(const half_integer<I>& x)
{
  return x.abs();
}

} // namespace alps


#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
namespace alps {
#endif

template <class I>
inline bool operator==(double x, const alps::half_integer<I>& y)
{ return y == x; }

template <class I>
inline bool operator!=(double x, const alps::half_integer<I>& y)
{ return y != x; }

template <class I>
inline bool operator<(double x, const alps::half_integer<I>& y)
{ return y > x; }

template <class I>
inline bool operator>(double x, const alps::half_integer<I>& y)
{ return y < x; }

template <class I>
inline bool operator<=(double x, const alps::half_integer<I>& y)
{ return y >= x; }

template <class I>
inline bool operator>=(double x, const alps::half_integer<I>& y)
{ return y <= x; }

template <class I>
inline
alps::half_integer<I> operator+(double x, const alps::half_integer<I>& y)
{ return y + x; }

template <class I>
inline
alps::half_integer<I> operator-(double x, const alps::half_integer<I>& y)
{ return - y + x; }

template <class I>
inline
std::ostream& operator<<(std::ostream& os, const alps::half_integer<I>& x)
{
  if (x == alps::half_integer<I>::max BOOST_PREVENT_MACRO_SUBSTITUTION ())
    return os << "infinity";
  else if (std::numeric_limits<I>::is_signed &&
           x == alps::half_integer<I>::min BOOST_PREVENT_MACRO_SUBSTITUTION ())
    return os << "-infinity";
  else if (x.get_twice()%2 ==0)
    return os << x.get_twice()/2;
  return os << x.get_twice() << "/2";
}

template <class I>
inline std::istream& operator>>(std::istream& is, alps::half_integer<I>& x)
{
  I nominator;
  is >> nominator;
  char c;
  is >> c;
  if ( is && c=='/') {
    is >> c;
    if (c!='2') {
      is.putback(c);
      is.putback('/');
      x.set_half(2*nominator);
    }
    x.set_half(nominator);
  }
  else {
    if (is)
      is.putback(c);
    x.set_half(2*nominator);
  }
  is.clear();
  return is;
}

#ifndef BOOST_NO_OPERATORS_IN_NAMESPACE
} // end namespace alps
#endif

#endif
