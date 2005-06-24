/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2005 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_UNIFORM_ON_SPHERE_N_H
#define ALPS_UNIFORM_ON_SPHERE_N_H

#include <boost/random/uniform_on_sphere.hpp>
#include <boost/random/uniform_real.hpp>
#include <cmath> // for std::sqrt
#include <vector>

namespace alps {

template<int N, class RealType = double, class Cont = std::vector<RealType> >
class uniform_on_sphere_n;

// generic version : wrapper on boost::uniform_on_sphere<>

template<int N, class RealType, class Cont>
class uniform_on_sphere_n
{
private:
  typedef boost::uniform_on_sphere<RealType, Cont> base_type;

public:
  typedef typename base_type::input_type input_type;
  typedef typename base_type::result_type result_type;

  BOOST_STATIC_CONSTANT(int, dim = N);

  uniform_on_sphere_n() : base_(dim) { }

  void reset() { base_.reset(); }

  template<class Engine>
  const result_type& operator()(Engine& eng) { return base_(eng); }

#if !defined(BOOST_NO_OPERATORS_IN_NAMESPACE) && !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const uniform_on_sphere_n& sd)
  {
    os << sd.base_;
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is, uniform_on_sphere_n& sd)
  {
    is >> sd.base_;
    return is;
  }
#endif

private:
  base_type base_;
};

// specialized version for N = 1, 2, and 3

template<class RealType, class Cont>
class uniform_on_sphere_n<1, RealType, Cont>
{
public:
  typedef RealType input_type;
  typedef Cont result_type;
  
  BOOST_STATIC_CONSTANT(int, dim = 1);

  uniform_on_sphere_n() : real_(0,1), container_(dim) { }

  void reset() {}

  template<class Engine>
  const result_type& operator()(Engine& eng)
  {
    container_[0] = (real_(eng) < 0.5) ? RealType(1) : RealType(-1);
    return container_;
  }

#if !defined(BOOST_NO_OPERATORS_IN_NAMESPACE) && !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const uniform_on_sphere_n&)
  {
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is, uniform_on_sphere_n& sd)
  {
    sd.container_.resize(sd.dim);
    return is;
  }
#endif

private:
  boost::uniform_real<RealType> real_;
  result_type container_;
};

template<class RealType, class Cont>
class uniform_on_sphere_n<2, RealType, Cont>
{
public:
  typedef RealType input_type;
  typedef Cont result_type;
  
  BOOST_STATIC_CONSTANT(int, dim = 2);

  uniform_on_sphere_n() : real_(-1,1), container_(dim) { }

  void reset() {}

  template<class Engine>
  const result_type& operator()(Engine& eng)
  {
    RealType v1, v2, s;
    do {
      v1 = real_(eng); // (-1..1)
      v2 = real_(eng); // (-1..1)
      s = v1 * v1 + v2 * v2;
    } while (s > 1);
#ifndef BOOST_NO_STDC_NAMESPACE
    using std::sqrt;
#endif
    const RealType a = 1.0 / std::sqrt(s);
    container_[0] = a * v1;
    container_[1] = a * v2;
    return container_;
  }

#if !defined(BOOST_NO_OPERATORS_IN_NAMESPACE) && !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const uniform_on_sphere_n&)
  {
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is, uniform_on_sphere_n& sd)
  {
    sd.container_.resize(sd.dim);
    return is;
  }
#endif

private:
  boost::uniform_real<RealType> real_;
  result_type container_;
};

template<class RealType, class Cont>
class uniform_on_sphere_n<3, RealType, Cont>
{
public:
  typedef RealType input_type;
  typedef Cont result_type;
  
  BOOST_STATIC_CONSTANT(int, dim = 3);

  uniform_on_sphere_n() : real_(-1,1), container_(dim) { }

  void reset() {}

  template<class Engine>
  const result_type& operator()(Engine& eng)
  {
    RealType v1, v2, s;
    do {
      v1 = real_(eng); // (-1..1)
      v2 = real_(eng); // (-1..1)
      s = v1 * v1 + v2 * v2;
    } while (s > 1);
#ifndef BOOST_NO_STDC_NAMESPACE
    using std::sqrt;
#endif
    const RealType a = 2 * std::sqrt(1 - s);
    container_[0] = a * v1;
    container_[1] = a * v2;
    container_[2] = 2 * s - 1;
    return container_;
  }

#if !defined(BOOST_NO_OPERATORS_IN_NAMESPACE) && !defined(BOOST_NO_MEMBER_TEMPLATE_FRIENDS)
  template<class CharT, class Traits>
  friend std::basic_ostream<CharT,Traits>&
  operator<<(std::basic_ostream<CharT,Traits>& os, const uniform_on_sphere_n&)
  {
    return os;
  }

  template<class CharT, class Traits>
  friend std::basic_istream<CharT,Traits>&
  operator>>(std::basic_istream<CharT,Traits>& is, uniform_on_sphere_n& sd)
  {
    sd.container_.resize(sd.dim);
    return is;
  }
#endif

private:
  boost::uniform_real<RealType> real_;
  result_type container_;
};

} // end namespace alps

#endif // ALPS_UNIFORM_ON_SPHERE_N_H
