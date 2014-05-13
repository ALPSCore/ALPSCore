/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_UTILITY_NUMERIC_CAST_HPP
#define ALPS_UTILITY_NUMERIC_CAST_HPP

#include <alps/numeric/real.hpp>
#include <vector>
#include <valarray>


namespace alps {

namespace detail {

template <class T, class U>
struct numeric_cast 
{
  typedef T return_type;
  static return_type cast(U const& u) { return static_cast<T>(u);}
};

template <class T>
struct numeric_cast<T,T> 
{
  typedef T const& return_type;
  static return_type cast(T const& u) { return u;}
};

template <class T, class U>
struct numeric_cast<T,std::complex<U> >
{
  typedef T return_type;
  static return_type cast( std::complex<U> const& u) {using alps::numeric::real; return real(u);}
};

template <class T, class U >
struct numeric_cast <std::complex<T>,std::complex<U> >
{
  typedef std::complex<T> return_type;
  static return_type cast(U const& u) { return static_cast<return_type>(u);}
};

template <class T>
struct numeric_cast<std::complex<T>,std::complex<T> > 
{
  typedef std::complex<T> const& return_type;
  static return_type cast(return_type u) { return u;}
};


template <class T, class U>
struct numeric_cast<std::valarray<T>,std::valarray<U> > 
{
  typedef std::valarray<T> return_type;
  
  static return_type cast(std::valarray<U> const& x) {
    return_type res(x.size());
    for (std::size_t i=0; i<x.size();++i)
      res[i]=numeric_cast<T,U>::cast(x[i]);
    return res;
  }
};

template <class T>
struct numeric_cast<std::valarray<T>,std::valarray<T> > 
{
  typedef std::valarray<T> const& return_type;
  
  static return_type cast(return_type u) { return u;}
};


template <class T, class U>
struct numeric_cast<std::vector<T>,std::vector<U> >
{ 
  typedef std::vector<T> return_type;

  static return_type cast(std::vector<U> const& x)
  { 
    return_type res;
    res.reserve(x.size());
    for (typename std::vector<U>::const_iterator it = x.begin(); it != x.end() ; ++it)
      res.push_back(numeric_cast<T,U>::cast(*it));
    return res;
  }
};


template <class T>
struct numeric_cast<std::vector<T>,std::vector<T> >
{ 
  typedef std::vector<T> const& return_type;

  static return_type cast(return_type u) { return u;}
};


}

template <class T, class U> 
inline typename detail::numeric_cast<T,U>::return_type numeric_cast(U const& u)
{
  return detail::numeric_cast<T,U>::cast(u);
}



} // end namespace alps

#endif // ALPS_UTILITY_NUMERIC_CAST_HPP
