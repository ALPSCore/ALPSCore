/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
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

#ifndef ALPS_ALEA_OBSVALUE_H
#define ALPS_ALEA_OBSVALUE_H

#include <alps/alea/convergence.hpp>
#include <alps/type_traits/slice.hpp>
#include <alps/type_traits/average_type.hpp>
#include <alps/type_traits/type_tag.hpp>
#include <alps/config.h>

#include <boost/config.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/lambda/lambda.hpp>

#include <complex>
#include <cstddef>
#include <vector>
#include <valarray>

namespace boost { 
namespace lambda {
  
template<class Act, class T> 
struct plain_return_type_2<arithmetic_action<Act>, std::valarray<T>, std::valarray<T> > {
  typedef std::valarray<T> type;
};

template<class Act, class T, class U> 
struct plain_return_type_2<arithmetic_action<Act>, std::valarray<T>, U> {
  typedef std::valarray<T> type;
};

template<class Act, class T, class U> 
struct plain_return_type_2<arithmetic_action<Act>, U, std::valarray<T> > {
  typedef std::valarray<T> type;
};

}
}

namespace alps {

namespace alea {

  typedef double count_type;

}

template <class T>
void update_max(T& lhs, T const& rhs)
{
  for (typename slice_index<T>::type it = slices(lhs).first; 
       it < slices(lhs).second && it < slices(rhs).second; ++it)
    if (slice_value(lhs,it) < slice_value(rhs,it))
      slice_value(lhs,it) = slice_value(rhs,it);
}

template <class T>
void update_min(T& lhs, T const& rhs)
{
  for (typename slice_index<T>::type it = slices(lhs).first; 
       it < slices(lhs).second && it < slices(rhs).second; ++it)
    if (slice_value(rhs,it) < slice_value(lhs,it))
      slice_value(lhs,it) = slice_value(rhs,it);
}

template <class T>
struct obs_value_traits
{
  BOOST_STATIC_CONSTANT( uint32_t, magic_id = type_tag<T>::value);

  template <class X>
  static inline X check_divide(const X& a,const X& b) 
    {
      return (b==0 && a==0? 1. : a/b); 
    }
    
  static void fix_negative(T& x) { if (x<0.) x=0.;}

  /* resize a to the lenth size */
  template <class X, class Y> static void resize_same_as(X&,const Y&) {}
  template <class X, class Y> static void copy(X& x,const Y& y) {x=y;}
  template <class X> static std::size_t size(const X&) { return 1;}
  
  template <class X>
  static inline X outer_product(X a, X b) {
    return a*b;
  }

  template <class X> static T convert(X x) { return static_cast<T>(x);}
};

template <class T>
struct obs_value_traits<std::complex<T> >
{
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = type_tag<T>::value);

  static void fix_negative(T& x) { if (std::real(x)<0. || std::imag(x)<0.) x=0.;}

  template <class X>
  static inline X check_divide(const X& a,const X& b)
    {
      return (b==0. && a==0. ? 1. : a/b); 
    }

  /** resize a to the lenth size */
  template <class X, class Y> static void resize_same_as(X&,const Y&) {}
  template <class X, class Y> static void copy(X& x,const Y& y) {x=y;}
  template <class X> static std::size_t size(const X&) { return 1;}

  template <class X>
  static inline X outer_product(X a, X b) {
    return std::conj(a)*b;
  }

  template <class X> static T convert(X x) { return static_cast<T>(x);}
};

#ifdef ALPS_HAVE_VALARRAY
template <class T>
struct obs_value_traits<std::valarray<T> >
{
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = 256+type_tag<T>::value);

  static void fix_negative(std::valarray<T>& a) 
  {
    for(int32_t i=0;i!=(int32_t)a.size();++i)
      obs_value_traits<typename element_type<T>::type>::fix_negative(a[i]);
  }

  template <class X>
  static inline X check_divide(const X& a,const X& b) 
  {
    X retval;
    resize_same_as(retval,b);
    for(int32_t i(0);i<(int32_t)b.size();++i)
      retval[i] = obs_value_traits<typename element_type<X>::type>::check_divide(a[i],b[i]);
    return retval;
  }

  /** resize a to given size */
  template <class X, class Y> static void resize_same_as(X& a, const Y& y) {a.resize(y.size());}
  template <class X, class Y> static void copy(X& x,const Y& y) {x.resize(y.size()); for (int i=0;i<(int)y.size();++i) x[i]=y[i];}
  template <class X> static std::size_t size(const X& a) { return a.size();}
  template <class X> static void resize(X& a, std::size_t s) {a.resize(s);}

  template <class X>
  static boost::numeric::ublas::matrix<typename average_type<T>::type> outer_product(X a, X b) 
  {
    boost::numeric::ublas::vector<typename average_type<T>::type> vec1(a.size()), vec2(b.size());
    for (int i=0; i<a.size(); ++i)
      vec1[i] = a[i];
    for (int i=0; i<b.size(); ++i)
      vec2[i] = b[i];
    return boost::numeric::ublas::outer_prod(vec1, vec2);

  }
 
  static std::valarray<T> const& convert(const std::valarray<T>& x)
  {
    return x;
  } 
  
  template <class X> static std::valarray<T> convert(const std::valarray<X>& x) 
  { 
    std::valarray<T> res;
    copy(res,x);
    return res;
  }

};

#endif

template <class T>
struct obs_value_traits<std::vector<T> >
{
  template <class X, class Y> static void resize_same_as(X& a, const Y& y) {a.resize(y.size());}
  template <class X, class Y> static void copy(X& x,const Y& y) {x.resize(y.size()); for (int i=0;i<y.size();++i) x[i]=y[i];}
  template <class X> static std::size_t size(const X& a) { return a.size();}
  template <class X> static void resize(X& a, std::size_t s) {a.resize(s);}

  static std::vector<T> const& convert(const std::vector<T>& x)
  {
    return x;
  } 
  
  template <class X> static std::vector<T> convert(const std::vector<X>& x) 
  { 
    std::vector<T> res;
    copy(res,x);
    return res;
  }


};

} // end namespace alps

#endif // ALPS_ALEA_OBSVALUE_H
