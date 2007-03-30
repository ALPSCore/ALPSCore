/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2006 by Matthias Troyer <troyer@comp-phys.org>,
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

#include <alps/config.h>
#include <alps/typetraits.h>
#include <boost/lexical_cast.hpp>
#include <boost/limits.hpp>
#include <boost/lambda/lambda.hpp>
#include <cstddef>
#include <vector>
#include <complex>
#include <valarray>
#include <boost/numeric/ublas/matrix.hpp>

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

enum error_convergence {CONVERGED, MAYBE_CONVERGED, NOT_CONVERGED};

inline std::string convergence_to_text(int c)
{
  return (c==CONVERGED ? "yes" : c==MAYBE_CONVERGED ? "maybe" : c==NOT_CONVERGED ? "no" : "");
}

template <class T>
struct obs_value_traits
{
  typedef T value_type; 
  typedef T element_type;
  typedef double count_type;
  typedef unsigned int index_type;
  typedef double time_type;
  typedef time_type time_element_type;
  typedef typename type_traits<T>::average_t result_type;
  typedef int convergence_type;
  typedef typename type_traits<T>::average_t covariance_type;
  typedef int slice_iterator;
  BOOST_STATIC_CONSTANT( uint32_t, magic_id = type_traits<T>::type_tag);
  typedef uint64_t size_type;
  BOOST_STATIC_CONSTANT( bool, array_valued = false);
  typedef std::string label_type;

  template <class X>
  static inline void check_for_max(T& a,const X& b) { if (b>a) a=b;}
  template <class X>
  static inline void check_for_min(T& a,const X& b) { if (b<a) a=b;}

  static T max() {return std::numeric_limits<T>::max();}
  static T min() {return -std::numeric_limits<T>::max();}
  static T epsilon() { return std::numeric_limits<T>::epsilon();}

  static inline time_type t_max() {return std::numeric_limits<time_type>::max();}

  static inline value_type check_divide(const result_type& a,const result_type& b) 
    {
      return (b==0 && a==0? 1. : a/b); 
    }
    
  static void fix_negative(value_type& x) { if (x<0.) x=0.;}

  /* resize a to the lenth size */
  template <class X, class Y> static void resize_same_as(X&,const Y&) {}
  template <class X, class Y> static void copy(X& x,const Y& y) {x=y;}
  template <class X> static std::size_t size(const X&) { return 1;}
  
  static inline covariance_type outer_product(result_type a, result_type b) {
    return a*b;
  }

  template <class X> static T convert(X x) { return static_cast<T>(x);}
  static slice_iterator slice_begin(const value_type&) { return 0;}
  static slice_iterator slice_end(const value_type&) { return 1;}
  static std::string slice_name(const value_type& ,slice_iterator) { return ""; }
  static element_type slice_value(const value_type& x, int) { return x;}
  static element_type& slice_value(value_type& x, int) { return x;}

};

/*
template <class DST> struct obs_value_cast
{
  DST const& operator()(DST const& x) { return x;}

  template <class SRC> 
  DST operator()(const SRC& s) 
  {
    return obs_value_traits<DST>::convert(s);
  }

template <class T> T const& obs_value_cast(const T& s, constT&=T()) 
{
  return s;
}


};
*/

template <class T>
struct obs_value_traits<std::complex<T> >
{
  typedef std::complex<T> value_type;
  typedef std::complex<T> element_type;
  typedef double count_type;
  typedef unsigned int index_type;
  typedef double time_type;
  typedef time_type time_element_type;
  typedef uint32_t size_type;
  typedef typename type_traits<T>::average_t result_type;
  typedef int convergence_type;
  typedef typename type_traits<T>::average_t covariance_type;  
  typedef int slice_iterator;
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = type_traits<T>::type_tag);
  BOOST_STATIC_CONSTANT(bool, array_valued=false);
  typedef std::string label_type;

  template <class X> static inline void check_for_max(std::complex<T>&,const X&) {}
  template <class X> static inline void check_for_min(std::complex<T>&,const X& b) {}

  static std::complex<T> max() { return  (std::numeric_limits<T>::max(),std::numeric_limits<T>::max());}
  static std::complex<T> min() { return  -max();}
  static T epsilon() { return std::numeric_limits<T>::epsilon();}

  static inline time_type t_max() { return  std::numeric_limits<time_type>::max();}

  static void fix_negative(value_type& x) { if (std::real(x)<0. || std::imag(x)<0.) x=0.;}

  static inline T check_divide(const result_type& a,const result_type& b)
    {
      return (b==0 && a==0 ? 1. : a/b); 
    }

  /** resize a to the lenth size */
  template <class X, class Y> static void resize_same_as(X&,const Y&) {}
  template <class X, class Y> static void copy(X& x,const Y& y) {x=y;}
  template <class X> static std::size_t size(const X&) { return 1;}

  static inline covariance_type outer_product(result_type a, result_type b) {
    return std::conj(a)*b;
  }

  template <class X> static T convert(X x) { return static_cast<T>(x);}
  static slice_iterator slice_begin(const value_type&) { return 0;}
  static slice_iterator slice_end(const value_type&) { return 1;}
  static std::string slice_name(const value_type& ,slice_iterator) { return ""; }
  static element_type slice_value(const value_type& x, int) { return x;}
  static element_type& slice_value(value_type& x, int) { return x;}
};

#ifdef ALPS_HAVE_VALARRAY
template <class T>
struct obs_value_traits<std::valarray<T> >
{
  typedef std::valarray<T> value_type;
  typedef T element_type;
  typedef double count_type;
  typedef unsigned int index_type;
  typedef std::size_t size_type;
  typedef std::valarray<double> time_type;
  typedef double time_element_type;
  BOOST_STATIC_CONSTANT(bool, array_valued = true);
  
  typedef std::valarray<typename type_traits<T>::average_t> result_type;
  typedef std::valarray<int> convergence_type;
  typedef typename boost::numeric::ublas::matrix<typename type_traits<T>::average_t> covariance_type;  
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = 256+type_traits<T>::type_tag);
  typedef std::vector<std::string> label_type;

  template <class X> static inline void check_for_max(std::valarray<T>& a,const std::valarray<X>& b) 
  {
    for(int32_t i=0;i!=(int32_t)a.size();++i)
      obs_value_traits<T>::check_for_max(a[i],b[i]);
  }   

  template <class X> static inline void check_for_max(std::valarray<T>& a,const X& b) 
  {
    for(int32_t i=0;i!=(int32_t)a.size();++i)
      obs_value_traits<T>::check_for_max(a[i],b);
  }   
 
  template <class X> static inline void check_for_min(std::valarray<T>& a,const std::valarray<X>& b) 
  {
    for(int32_t i=0;i!=(int32_t)a.size();++i)
      obs_value_traits<T>::check_for_min(a[i],b[i]);
  }    

  template <class X> static inline void check_for_min(std::valarray<T>& a,const X& b) 
  {
    for(int32_t i=0;i!=(int32_t)a.size();++i)
      obs_value_traits<T>::check_for_min(a[i],b);
  }    

  static void fix_negative(value_type& a) 
  {
    for(int32_t i=0;i!=(int32_t)a.size();++i)
      obs_value_traits<element_type>::fix_negative(a[i]);
  }
  static element_type min() {return obs_value_traits<T>::min();}
  static element_type max() {return obs_value_traits<T>::max();}
  static element_type epsilon() { return obs_value_traits<T>::epsilon();}

  static time_element_type t_max() {return obs_value_traits<T>::max();}

  static inline time_type check_divide(const result_type& a,const result_type& b) 
  {
    time_type retval;
    resize_same_as(retval,b);
    for(int32_t i(0);i<(int32_t)b.size();++i)
      retval[i] = obs_value_traits<element_type>::check_divide(a[i],b[i]);
    return retval;
  }

  /** resize a to given size */
  template <class X, class Y> static void resize_same_as(X& a, const Y& y) {a.resize(y.size());}
  template <class X, class Y> static void copy(X& x,const Y& y) {x.resize(y.size()); for (int i=0;i<(int)y.size();++i) x[i]=y[i];}
  template <class X> static std::size_t size(const X& a) { return a.size();}
  template <class X> static void resize(X& a, std::size_t s) {a.resize(s);}

  static covariance_type outer_product(result_type a, result_type b) 
  {
    boost::numeric::ublas::vector<typename type_traits<T>::average_t> vec1(a.size()), vec2(b.size());
    for (int i=0; i<a.size(); ++i)
      vec1[i] = a[i];
    for (int i=0; i<b.size(); ++i)
      vec2[i] = b[i];
    return boost::numeric::ublas::outer_prod(vec1, vec2);

  }

  typedef uint32_t slice_iterator;
  static slice_iterator slice_begin(const value_type&) { return 0;}
  static slice_iterator slice_end(const value_type& x) { return x.size();}
  static std::string slice_name(const value_type& ,slice_iterator i) 
    { return boost::lexical_cast<std::string,int>(i); }
  static element_type slice_value(const value_type& x, slice_iterator i) { return x[i];}
  static element_type& slice_value(value_type& x, slice_iterator i) { return  x[i];}
 
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
  typedef std::vector<T> value_type;
  typedef T element_type;
  template <class X, class Y> static void resize_same_as(X& a, const Y& y) {a.resize(y.size());}
  template <class X, class Y> static void copy(X& x,const Y& y) {x.resize(y.size()); for (int i=0;i<y.size();++i) x[i]=y[i];}
  template <class X> static std::size_t size(const X& a) { return a.size();}
  template <class X> static void resize(X& a, std::size_t s) {a.resize(s);}

  typedef uint32_t slice_iterator;
  static slice_iterator slice_begin(const value_type&) { return 0;}
  static slice_iterator slice_end(const value_type& x) { return x.size();}
  static std::string slice_name(const value_type& ,slice_iterator i) 
    { return boost::lexical_cast<std::string,int>(i); }
  static element_type slice_value(const value_type& x, slice_iterator i) { return (i<x.size()) ? x[i] : element_type();}
  static element_type& slice_value(value_type& x, slice_iterator i) { return (i<x.size()) ? x[i] : element_type();}

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

template <class T,class I>
struct obs_value_slice
{                               
  BOOST_STATIC_CONSTANT(bool, sliceable=false);
};                                            

#ifdef ALPS_HAVE_VALARRAY
template <class T, class I>                           
struct obs_value_slice<std::valarray<T>,I>  
{                                           
  BOOST_STATIC_CONSTANT(bool, sliceable=true);       
  typedef T value_type;                  
  typedef T result_type;                
  typedef const std::valarray<T>& first_argument_type;
  typedef I second_argument_type;            
  T operator()(const std::valarray<T>& x, const I i) const { return x[i];}
};                                             
#endif

} // end namespace alps

#endif // ALPS_ALEA_OBSVALUE_H
