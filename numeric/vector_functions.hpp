/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Ping Nang Ma <pingnang@itp.phys.ethz.ch>,
*                            Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#ifndef ALPS_NUMERIC_VECTOR_FUNCTIONS_HPP
#define ALPS_NUMERIC_VECTOR_FUNCTIONS_HPP


#include <alps/numeric/special_functions.hpp>

#include <alps/boost/accumulators/numeric/functional/vector.hpp>
#include <alps/boost/accumulators/numeric/functional.hpp>

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
    inline std::vector<T> operator/(T const & scalar, std::vector<T> vector) 
    { 
      std::vector<T> res; 
      res.reserve(vector.size()); 
      for (typename std::vector<T>::iterator it=vector.begin(); it != vector.end(); ++it) 
      { 
        res.push_back(scalar / (*it)); 
      } 
      return res; 
    } 


    

    // include important functions for vectors
    #define IMPLEMENT_ALPS_VECTOR_FUNCTION(LIB_HEADER,FUNCTION_NAME) \
    template<class T> \
    std::vector<T> FUNCTION_NAME(std::vector<T> vec) \
    { \
      using LIB_HEADER::FUNCTION_NAME; \
      std::transform(vec.begin(), vec.end(), vec.begin(), static_cast<T (*)(T)>(&FUNCTION_NAME)); \
      return vec; \
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
    template<class T> \
    static std::vector<T> FUNCTION_NAME(std::vector<T> vec, T index) \
    { \
      using LIB_HEADER::FUNCTION_NAME; \
      using boost::lambda::_1; \
      using boost::lambda::bind; \
      std::transform(vec.begin(), vec.end(), vec.begin(), bind<T>(static_cast<T (*)(T, T)>(&FUNCTION_NAME), _1, index)); \
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




