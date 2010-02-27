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
#include <alps/type_traits/is_sequence.hpp>
#include <alps/type_traits/type_tag.hpp>
#include <alps/config.h>

#include <boost/config.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/mpl/and.hpp>
#include <boost/mpl/or.hpp>

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
inline typename boost::disable_if<is_sequence<T>,void>::type
set_negative_0(T& x)
{
  if (x<T()) 
    x=T();
}

template <class T>
inline void set_negative_0(std::complex<T>& x)
{ 
  if (std::real(x)<0. || std::imag(x)<0.) 
    x=0.;
}

template <class T>
inline typename boost::enable_if<is_sequence<T>,void>::type
set_negative_0(T& a) 
{
  for(std::size_t i=0; i!=a.size(); ++i)
    set_negative_0(a[i]);
}





template <class T>
inline typename boost::disable_if<is_sequence<T>,T>::type
checked_divide(const T& a,const T& b) 
{
  return (b==T() && a==T()? 1. : a/b); 
}

template <class T>
inline typename boost::enable_if<is_sequence<T>,T>::type
checked_divide(T a,const T& b) 
{
  for(std::size_t i=0;i<b.size();++i)
    a[i] = checked_divide(a[i],b[i]);
  return a;
}


template <class X, class Y> 
inline void assign(X& x,const Y& y) 
{
  x=y;
}

template <class X, class Y> 
inline void assign(std::valarray<X>& x, std::valarray<Y> const& y) 
{
  x.resize(y.size()); 
  for (std::size_t i=0;i<y.size();++i) 
    x[i]=y[i];
}

template <class X> 
inline void assign(std::valarray<X>& x, std::valarray<X> const& y) 
{
  x.resize(y.size()); 
  x=y;
}






template <class X, class Y> 
inline typename boost::disable_if<boost::mpl::or_<is_sequence<X>,is_sequence<Y> >,void>::type
resize_same_as(X&, const Y&) {}

template <class X, class Y> 
inline typename boost::enable_if<boost::mpl::and_<is_sequence<X>,is_sequence<Y> >,void>::type
resize_same_as(X& a, const Y& y) 
{
  a.resize(y.size());
}





template <class T>
inline typename boost::disable_if<is_sequence<T>,std::size_t>::type
size(T const& a) 
{
  return 1;
}

template <class T>
inline typename boost::enable_if<is_sequence<T>,std::size_t>::type
size(T const& a) 
{
  return a.size();
}




template <class T>
struct obs_value_traits
{
  template <class X>
  static inline X outer_product(X a, X b) {
    return a*b;
  }

  template <class X> static T convert(X x) { return static_cast<T>(x);}
};

template <class T>
struct obs_value_traits<std::complex<T> >
{
  template <class X>
  static inline X outer_product(X a, X b) {
    return std::conj(a)*b;
  }

  template <class X> static T convert(X x) { return static_cast<T>(x);}
};

template <class T>
struct obs_value_traits<std::valarray<T> >
{
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
    std::valarray<T> res(x.size());
    for (std::size_t i=0; i<x.size();++i)
      res[i]=x[i];
    return res;
  }

};


template <class T>
struct obs_value_traits<std::vector<T> >
{
  static std::vector<T> const& convert(const std::vector<T>& x)
  {
    return x;
  } 
  
  template <class X> static std::vector<T> convert(const std::vector<X>& x) 
  { 
    return std::vector<T>(x.begin(),x.end());
  }


};

} // end namespace alps

#endif // ALPS_ALEA_OBSVALUE_H
