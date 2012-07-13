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
