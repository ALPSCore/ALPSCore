/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#ifndef ALPS_NUMERIC_VECTOR_FUNCTIONS_HPP
#define ALPS_NUMERIC_VECTOR_FUNCTIONS_HPP


#include <alps/numeric/special_functions.hpp>

#include <boost/accumulators/numeric/functional/vector.hpp> 
#include <boost/accumulators/numeric/functional.hpp> 

#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>

namespace alps { 
  namespace numeric {

    // include ( + (add) , - (subtract, negation) , * (multiply) , / (divide) ) operators for vectors from boost accumulator library (developed by Eric Niebier)
    using namespace boost::numeric::operators;

    // include ( + (positivity) )
    template<class T>
    inline std::vector<T>& operator+(std::vector<T>& vec)  {  return vec;  }


    // include ( + , - , * , / vector-scalar operations)
    #define IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(OPERATOR_NAME,OPERATOR) \
    template<class T> \
    inline std::vector<T> OPERATOR_NAME(std::vector<T> vector, T const & scalar) \
    { \
      std::vector<T> res; \
      res.reserve(vector.size()); \
      for (typename std::vector<T>::iterator it=vector.begin(); it != vector.end(); ++it) \
      { \
        res.push_back(((*it) OPERATOR scalar)); \
      } \
      return res; \
    } \
    \
    template<class T> \
    inline std::vector<T> OPERATOR_NAME(T const & scalar, std::vector<T> vector) \
    { \
      std::vector<T> res; \
      res.reserve(vector.size()); \
      for (typename std::vector<T>::iterator it=vector.begin(); it != vector.end(); ++it) \
      { \
        res.push_back((scalar OPERATOR (*it))); \
      } \
      return res; \
    } 

    IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator+,+)
    IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator-,-)
    
// the following two may not be defined as they are already defined by Boost
//    IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator*,*)
//    IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator/,/)

    template<class T> 
    inline std::vector<T> operator/(T const & scalar, std::vector<T> const& vector) 
    { 
      std::vector<T> res; 
      res.reserve(vector.size()); 
      for (typename std::vector<T>::const_iterator it=vector.begin(); it != vector.end(); ++it) 
      { 
        res.push_back(scalar / (*it)); 
      } 
      return res; 
    } 


    // fix for old xlc compilers
    #define IMPLEMENT_ALPS_VECTOR_FUNCTION(LIB_HEADER, FUNCTION_NAME)                                        \
        namespace detail {                                                                                    \
            template<typename T> struct FUNCTION_NAME ## _VECTOR_OP_HELPER {                                        \
                T operator() (T arg) {                                                                        \
                    using LIB_HEADER :: FUNCTION_NAME;                                                        \
                    return FUNCTION_NAME (arg);                                                                \
                }                                                                                            \
            };                                                                                                \
        }                                                                                                    \
        template<typename T> std::vector<T> FUNCTION_NAME(std::vector<T> vec) {                                \
            std::transform(vec.begin(), vec.end(), vec.begin(), detail:: FUNCTION_NAME ## _VECTOR_OP_HELPER<T>());    \
            return vec;                                                                                        \
        }

    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,abs)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(alps::numeric,sq)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,sqrt)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(alps::numeric,cb)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(alps::numeric,cbrt)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,exp)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,log)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,sin)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,cos)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,tan)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,asin)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,acos)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,atan)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,sinh)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,cosh)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(std,tanh)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(boost::math,asinh)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(boost::math,acosh)
    IMPLEMENT_ALPS_VECTOR_FUNCTION(boost::math,atanh)

    #define IMPLEMENT_ALPS_VECTOR_FUNCTION2(LIB_HEADER,FUNCTION_NAME) \
    template<class T, class U> \
    static std::vector<T> FUNCTION_NAME(std::vector<T> vec, U index) \
    { \
      using LIB_HEADER::FUNCTION_NAME; \
      std::transform(vec.begin(), vec.end(), vec.begin(), boost::lambda::bind<T>(static_cast<T (*)(T, U)>(&FUNCTION_NAME), boost::lambda::_1, index)); \
      return vec; \
    }

    IMPLEMENT_ALPS_VECTOR_FUNCTION2(std,pow)

    template <class T>
    std::ostream& operator<< (std::ostream &out, std::vector<T> const & vec)
    {
      std::copy(vec.begin(),vec.end(),std::ostream_iterator<T>(out,"\t"));
      return out;
    }
  }
}

#endif // ALPS_NUMERIC_VECTOR_FUNCTIONS_HPP




