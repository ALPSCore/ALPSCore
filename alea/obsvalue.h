/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2003 by Matthias Troyer <troyer@comp-phys.org>,
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
#include <alps/multi_array.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/limits.hpp>
#include <cstddef>
#include <complex>

#ifdef ALPS_HAVE_VALARRAY
# include <valarray>
#endif

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
  typedef typename TypeTraits<T>::average_t result_type;
  typedef int convergence_type;
  typedef int slice_iterator;
  BOOST_STATIC_CONSTANT( uint32_t, magic_id = TypeTraits<T>::type_tag);
  typedef uint32_t size_type;
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
  
  template <class X> static T convert(X x) { return static_cast<T>(x);}
  static slice_iterator slice_begin(const value_type&) { return 0;}
  static slice_iterator slice_end(const value_type&) { return 1;}
  static std::string slice_name(const value_type& ,slice_iterator) { return ""; }
  static element_type slice_value(const value_type& x, int) { return x;}
  static element_type& slice_value(value_type& x, int) { return x;}

};

template <class DST,class SRC> DST obs_value_cast(const SRC& s) 
{
  return obs_value_traits<DST>::convert(s);
}

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
  typedef typename TypeTraits<T>::average_t result_type;
  typedef int convergence_type;
  typedef int slice_iterator;
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = TypeTraits<T>::type_tag);
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
  
  typedef std::valarray<typename TypeTraits<T>::average_t> result_type;
  typedef std::valarray<int> convergence_type;
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = 256+TypeTraits<T>::type_tag);
  typedef std::vector<std::string> label_type;

  template <class X> static inline void check_for_max(std::valarray<T>& a,const std::valarray<X>& b) 
  {
    for(int32_t i=0;i!=a.size();++i)
      obs_value_traits<T>::check_for_max(a[i],b[i]);
  }   

  template <class X> static inline void check_for_max(std::valarray<T>& a,const X& b) 
  {
    for(int32_t i=0;i!=a.size();++i)
      obs_value_traits<T>::check_for_max(a[i],b);
  }   
 
  template <class X> static inline void check_for_min(std::valarray<T>& a,const std::valarray<X>& b) 
  {
    for(int32_t i=0;i!=a.size();++i)
      obs_value_traits<T>::check_for_min(a[i],b[i]);
  }    

  template <class X> static inline void check_for_min(std::valarray<T>& a,const X& b) 
  {
    for(int32_t i=0;i!=a.size();++i)
      obs_value_traits<T>::check_for_min(a[i],b);
  }    

  static void fix_negative(value_type& a) 
  {
    for(int32_t i=0;i!=a.size();++i)
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
    for(int32_t i(0);i<b.size();++i)
      retval[i] = obs_value_traits<element_type>::check_divide(a[i],b[i]);
    return retval;
  }

  /** resize a to given size */
  template <class X, class Y> static void resize_same_as(X& a, const Y& y) {a.resize(y.size());}
  template <class X, class Y> static void copy(X& x,const Y& y) {x.resize(y.size()); for (int i=0;i<y.size();++i) x[i]=y[i];}
  template <class X> static std::size_t size(const X& a) { return a.size();}
  template <class X> static void resize(X& a, std::size_t s) {a.resize(s);}

  typedef uint32_t slice_iterator;
  static slice_iterator slice_begin(const value_type&) { return 0;}
  static slice_iterator slice_end(const value_type& x) { return x.size();}
  static std::string slice_name(const value_type& ,slice_iterator i) 
    { return boost::lexical_cast<std::string,int>(i); }
  static element_type slice_value(const value_type& x, slice_iterator i) { return x[i];}
  static element_type& slice_value(value_type& x, slice_iterator i) { return x[i];}
 
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
  static element_type slice_value(const value_type& x, slice_iterator i) { return x[i];}
  static element_type& slice_value(value_type& x, slice_iterator i) { return x[i];}
};

template<typename T, std::size_t NumDims, typename Allocator>
  struct obs_value_traits<alps::multi_array<T,NumDims,Allocator> >
{
  typedef alps::multi_array<T,NumDims,Allocator> value_type;
  typedef T element_type;
  typedef double count_type;
  typedef typename value_type::index index_type;
  typedef typename value_type::size_type size_type;
  typedef alps::multi_array<double,NumDims> time_type;
  typedef double time_element_type;
  BOOST_STATIC_CONSTANT(bool, array_valued = true);
  
  typedef alps::multi_array<typename TypeTraits<T>::average_t,NumDims> result_type;
  typedef alps::multi_array<int,NumDims> convergence_type;
  BOOST_STATIC_CONSTANT(uint32_t, magic_id = (1+NumDims)*256+TypeTraits<T>::type_tag);
  typedef alps::multi_array<std::string,NumDims> label_type;

  template <class X, class A> static inline void check_for_max(value_type& a,const alps::multi_array<X,NumDims,A>& b) 
  {
    typename value_type::iterator ait=a.begin();
    typename value_type::const_iterator bit=b.begin();
    for(;ait!=a.end() && bit!=b.end();++ait,++bit)
      obs_value_traits<T>::check_for_max(*ait,*bit);
    if (ait!=a.end() || bit!=b.end())
      boost::throw_exception(std::runtime_error("multi_arrays not of identical size in obs_value_traits::check_for_max"));
  }   
 
  template <class X, class A> static inline void check_for_min(value_type& a,const alps::multi_array<X,NumDims,A>& b) 
  {
    typename value_type::iterator ait=a.begin();
    typename value_type::const_iterator bit=b.begin();
    for(;ait!=a.end() && bit!=b.end();++ait,++bit)
      obs_value_traits<T>::check_for_min(*ait,*bit);
    if (ait!=a.end() || bit!=b.end())
      boost::throw_exception(std::runtime_error("multi_arrays not of identical size in obs_value_traits::check_for_min"));
  }    

  template <class X> static inline void check_for_max(value_type& a,const X& b) 
  {
    for(typename value_type::iterator ait=a.begin() ;ait!=a.end() ;++ait)
      obs_value_traits<T>::check_for_max(*ait,b);
  }   
 
  template <class X> static inline void check_for_min(value_type& a,const X& b) 
  {
    for(typename value_type::iterator ait=a.begin() ;ait!=a.end() ;++ait)
      obs_value_traits<T>::check_for_min(*ait,b);
  }    

  static element_type min() {return obs_value_traits<T>::min();}
  static element_type max() {return obs_value_traits<T>::max();}
  static element_type epsilon() { return obs_value_traits<T>::epsilon();}

  static time_element_type t_max() {return obs_value_traits<T>::max();}

  static inline time_type check_divide(const result_type& a,const result_type& b) 
  {
    time_type retval;
    resize_same_as(retval,b);

    typename value_type::const_iterator ait=a.begin();
    typename value_type::const_iterator bit=b.begin();
    typename time_type::iterator rit=retval.begin();

    for(;ait!=a.end() && bit!=b.end() && rit !=retval.end();++ait,++bit,++rit)
      *rit= obs_value_traits<element_type>::check_divide(*ait,*bit);
    if (ait!=a.end() || bit!=b.end() || rit !=retval.end())
      boost::throw_exception(std::runtime_error("multi_arrays not of identical size in obs_value_traits::check_divide"));
    return retval;
  }

  static void fix_negative(value_type& a) 
  {
    for(typename value_type::iterator it=a.begin();it!=a.end();++it)
      obs_value_traits<element_type>::fix_negative(*it);
  }


  template <class X, class AX, class Y, class AY> 
  static void resize_same_as(boost::multi_array<X,NumDims,AX>& x,
                             const boost::multi_array<Y,NumDims,AY>& y)
  {
    x=boost::multi_array<X,NumDims,AX>(
     std::vector<boost::multi_array_types::size_type>(
       y.shape(),y.shape()+y.num_dimensions()));
  }
  
  template <class X, class AX, class Y,class AY> 
  static void copy(boost::multi_array<X,NumDims,AX>& x,
                             const boost::multi_array<Y,NumDims,AY>& y)
  {
    resize_same_as(x,y);
    x=y;
  }

  template <class X> static size_type size(const X& a) { return a.num_elements();}

  struct slice_iterator {
    typedef boost::array<typename value_type::index,NumDims> index_type;
    
    bool operator==(const slice_iterator rhs) { return idx==rhs.idx;}
    
    slice_iterator() {} // IRIX MIPSpro compiler requires explicit
                        // definition of default constructor

    slice_iterator(const value_type& array, bool is_begin=true)
    {
      std::copy(array.index_bases(),array.index_bases()+NumDims,bases.begin());
      std::copy(array.shape(),array.shape()+NumDims,shape.begin());
      for (int i=0;i<NumDims;++i) {
        shape[i]+=bases[i];
        idx[i]= (is_begin ? bases[i] : shape[i]);
        }
    }
    
    const slice_iterator& operator++() {
      int dim=0;
      idx[dim]++;
      while(idx[dim]==shape[dim] && dim<NumDims-1) {
        idx[dim]=bases[dim];
        dim++;
        idx[dim]++;
      }
      return *this;
    }
    
    slice_iterator operator++(int) 
    {
      slice_iterator tmp(*this);
      operator++();
      return tmp;
    }
    
    std::string name() const 
    {
      std::string n;
      for (int i=0;i<NumDims;++i) {
        n += boost::lexical_cast<std::string,size_type>(idx[i]);
        if (i!=NumDims-1)
        n+=", ";
      }
      return n;
    }
    
    bool operator ==(const slice_iterator& rhs) const { return idx==rhs.idx;}
    bool operator !=(const slice_iterator& rhs) const { return idx!=rhs.idx;}
    const index_type& index() const { return idx;}
  private:
    index_type idx;
    index_type shape;
    index_type bases;
  };
  
  static slice_iterator slice_begin(const value_type& x) { return slice_iterator(x,true);}
  static slice_iterator slice_end(const value_type& x) { return slice_iterator(x,false);}
  static std::string slice_name(const value_type& ,slice_iterator i) { return i.name(); }
  static element_type slice_value(const value_type& x, slice_iterator i) { return x(i.index());}
  static element_type& slice_value(value_type& x, slice_iterator i) { return x(i.index());}

  template <class X, class A> 
  static alps::multi_array<T,NumDims,Allocator> convert(const alps::multi_array<X,NumDims,A>& x) 
  { 
    return alps::multi_array<T,NumDims,Allocator> (x);
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
  double operator()(const std::valarray<T>& x, const I i) const { return x[i];}
};                                             
#endif

template<typename T, std::size_t NumDims, typename Allocator, class I>
  struct obs_value_slice<alps::multi_array<T,NumDims,Allocator>,I>
{                                           
  BOOST_STATIC_CONSTANT(bool, sliceable=true);       
  typedef T value_type;                  
  typedef T result_type;                
  typedef const alps::multi_array<T,NumDims,Allocator>& first_argument_type;
  typedef I second_argument_type;            
  double operator()(const alps::multi_array<T,NumDims,Allocator>& x, const I& i) const { return x(i);}
};                                             

} // end namespace alps

#endif // ALPS_ALEA_OBSVALUE_H
