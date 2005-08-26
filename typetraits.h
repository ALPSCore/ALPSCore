/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

/// \file typetraits.h
/// \brief useful type traits
/// 
/// This header contains type traits, mainly for numeric types

#ifndef ALPS_TYPETRAITS_H
#define ALPS_TYPETRAITS_H

#include <alps/config.h>
#include <boost/limits.hpp>
#include <complex>
#include <string>

namespace alps {

/// the type to store a tag number for each type
typedef int32_t type_tag_t;

/// \brief A class to store useful traits for common types.  
///
///    Specializations exist for: 
///    - bool
///    - int8_t, int16_t, int32_t, int64_t
///    - uint8_t, uint16_t, uint32_t, uint64_t 
///    - float, double, long double,
///    - std::complex<float>, std::complex<double>, std::complex<long double>
///    - std::string.
template <class T>
struct type_traits
{
  /// is true if type_traits is specialized for a type
  BOOST_STATIC_CONSTANT(bool, is_specialized = false);
  /// \brief a numeric identifier for the type
  ///
  /// is -1 if \a is_specialized is false, and a nonnegative number otherwise
  BOOST_STATIC_CONSTANT(type_tag_t, type_tag = -1);
  /// \brief true for complex numbers
  BOOST_STATIC_CONSTANT(bool, is_complex = false);
  /// type to store the norm of a value, useful only for numeric types
  typedef T norm_t;
  /// type to store the averages of values, useful only for numeric types
  typedef T average_t;
  /// signed type corresponding to the type
  typedef T signed_t;
  /// unsigned type corresponding to the type
  typedef T unsigned_t;
  /// for complex types this gives the type of the real and imaginary parts
  typedef T real_t;              
  /// machine epsilon for the norm type
  static norm_t epsilon() { return std::numeric_limits<norm_t>::epsilon(); }
};

#define DEFINE_NUMERIC_TYPE_TRAITS(TYPE,TAG,NORMT,SIGNT,UNSIGNT,REALT,AVT) \
template<> struct type_traits< TYPE > {                               \
  BOOST_STATIC_CONSTANT(bool, is_specialized = true);      \
  BOOST_STATIC_CONSTANT(type_tag_t, type_tag=TAG);     \
  BOOST_STATIC_CONSTANT(bool, is_complex = (TAG>=3 && TAG <= 5)); \
  typedef NORMT norm_t;                                \
  typedef SIGNT signed_t;                                \
  typedef UNSIGNT unsigned_t;                                \
  typedef REALT real_t;                                \
  typedef AVT average_t;       \
  static norm_t epsilon() { return std::numeric_limits<norm_t>::epsilon(); } \
};

/// spezialization of type_traits for float
DEFINE_NUMERIC_TYPE_TRAITS(float,0,float,float,float,float,float)
/// spezialization of type_traits for double
DEFINE_NUMERIC_TYPE_TRAITS(double,1,double,double,double,double,double)
/// spezialization of type_traits for long double
DEFINE_NUMERIC_TYPE_TRAITS(long double,2,long double,long double, long double,long double,long double)
/// spezialization of type_traits for complex<float>
DEFINE_NUMERIC_TYPE_TRAITS(std::complex<float>,3,float,std::complex<float>,std::complex<float>,float,std::complex<float>)
/// spezialization of type_traits for complex<double>
DEFINE_NUMERIC_TYPE_TRAITS(std::complex<double>,4,double,std::complex<double>,std::complex<double>,double,std::complex<double>)
/// spezialization of type_traits for complex<long double>
DEFINE_NUMERIC_TYPE_TRAITS(std::complex<long double>,5,long double,std::complex<long double>,std::complex<long double>,long double,std::complex<long double>)
/// spezialization of type_traits for int16_t
DEFINE_NUMERIC_TYPE_TRAITS(int16_t,6,double,int16_t,uint16_t,int16_t,double)
/// spezialization of type_traits for int32_t
DEFINE_NUMERIC_TYPE_TRAITS(int32_t,7,double,int32_t,uint32_t,int32_t,double)
#ifndef BOOST_NO_INT64_T
/// spezialization of type_traits for int64_t
DEFINE_NUMERIC_TYPE_TRAITS(int64_t,8,double,int64_t,uint64_t,int64_t,double)
#endif
/// spezialization of type_traits for uint16_t
DEFINE_NUMERIC_TYPE_TRAITS(uint16_t,9,double,int16_t,uint16_t,uint16_t,double)
/// spezialization of type_traits for uint32_t
DEFINE_NUMERIC_TYPE_TRAITS(uint32_t,10,double,int32_t,uint32_t,uint32_t,double)
#ifndef BOOST_NO_INT64_T
/// spezialization of type_traits for uint64_t
DEFINE_NUMERIC_TYPE_TRAITS(uint64_t,11,double,int64_t,uint64_t,uint64_t,double)
#endif
/// spezialization of type_traits for int8_t
DEFINE_NUMERIC_TYPE_TRAITS(int8_t,12,double,int8_t,uint8_t,int8_t,double)
/// spezialization of type_traits for uint8_t
DEFINE_NUMERIC_TYPE_TRAITS(uint8_t,13,double,int8_t,uint8_t,uint8_t,double)
/// spezialization of type_traits for bool
DEFINE_NUMERIC_TYPE_TRAITS(bool,15,bool,bool,bool,bool,bool)
#undef DEFINE_NUMERIC_TYPE_TRAITS

/// spezialization of type_traits for std::string
template<>
struct type_traits<std::string>
{
  BOOST_STATIC_CONSTANT(bool, is_specialized = true);      
  BOOST_STATIC_CONSTANT(type_tag_t, type_tag=14);     
  BOOST_STATIC_CONSTANT(bool, is_complex = false);
};


} // namespace alps

#endif // ALPS_TYPETRAITS_H
