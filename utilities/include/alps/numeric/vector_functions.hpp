/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: nobinning.h 3520 2009-12-11 16:49:53Z gamperl $ */

#ifndef ALPS_NUMERIC_VECTOR_FUNCTIONS_HEADER
#define ALPS_NUMERIC_VECTOR_FUNCTIONS_HEADER


#include <alps/utilities/stacktrace.hpp>

#include <alps/numeric/inf.hpp>
#include <alps/numeric/special_functions.hpp>

#include <boost/accumulators/numeric/functional/vector.hpp> 
#include <boost/accumulators/numeric/functional.hpp> 

#include <boost/bind.hpp>
#include <boost/foreach.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/throw_exception.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace alps { 
    namespace numeric {

      //   // include ( + (add) , - (subtract, negation) , * (multiply) , / (divide) ) operators for vectors from boost accumulator library (developed by Eric Niebier)
      //   using namespace boost::numeric::operators;

      //   // include ( + (positivity) )
      //   template<class T>
      //   inline std::vector<T>& operator+(std::vector<T>& vec)  {  return vec;  }


      //   // include ( + , - , * , / vector-scalar operations)
      //   #define IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(OPERATOR_NAME,OPERATOR) \
      //   template<class T> \
      //   inline std::vector<T> OPERATOR_NAME(std::vector<T> vector, T const & scalar) \
      //   { \
      //     std::vector<T> res; \
      //     res.reserve(vector.size()); \
      //     for (typename std::vector<T>::iterator it=vector.begin(); it != vector.end(); ++it) \
      //     { \
      //       res.push_back(((*it) OPERATOR scalar)); \
      //     } \
      //     return res; \
      //   } \
      //   \
      //   template<class T> \
      //   inline std::vector<T> OPERATOR_NAME(T const & scalar, std::vector<T> vector) \
      //   { \
      //     std::vector<T> res; \
      //     res.reserve(vector.size()); \
      //     for (typename std::vector<T>::iterator it=vector.begin(); it != vector.end(); ++it) \
      //     { \
      //       res.push_back((scalar OPERATOR (*it))); \
      //     } \
      //     return res; \
      //   } 

      //   IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator+,+)
      //   IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator-,-)
        
      // // the following two may not be defined as they are already defined by Boost
      // //    IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator*,*)
      // //    IMPLEMENT_ALPS_VECTOR_SCALAR_OPERATION(operator/,/)

      //   template<class T> 
      //   inline std::vector<T> operator/(T const & scalar, std::vector<T> const& vector) 
      //   { 
      //     std::vector<T> res; 
      //     res.reserve(vector.size()); 
      //     for (typename std::vector<T>::const_iterator it=vector.begin(); it != vector.end(); ++it) 
      //     { 
      //       res.push_back(scalar / (*it)); 
      //     } 
      //     return res; 
      //   } 


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

        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,abs)
        IMPLEMENT_ALPS_VECTOR_FUNCTION(alps::numeric,sq)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,sqrt)
        IMPLEMENT_ALPS_VECTOR_FUNCTION(alps::numeric,cb)
        IMPLEMENT_ALPS_VECTOR_FUNCTION(alps::numeric,cbrt)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,exp)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,log)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,sin)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,cos)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,tan)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,asin)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,acos)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,atan)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,sinh)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,cosh)
        // IMPLEMENT_ALPS_VECTOR_FUNCTION(std,tanh)
        IMPLEMENT_ALPS_VECTOR_FUNCTION(boost::math,asinh)
        IMPLEMENT_ALPS_VECTOR_FUNCTION(boost::math,acosh)
        IMPLEMENT_ALPS_VECTOR_FUNCTION(boost::math,atanh)

        #undef IMPLEMENT_ALPS_VECTOR_FUNCTION

      //   #define IMPLEMENT_ALPS_VECTOR_FUNCTION2(LIB_HEADER,FUNCTION_NAME) \
      //   template<class T, class U> \
      //   static std::vector<T> FUNCTION_NAME(std::vector<T> vec, U index) \
      //   { \
      //     using LIB_HEADER::FUNCTION_NAME; \
      //     std::transform(vec.begin(), vec.end(), vec.begin(), boost::lambda::bind<T>(static_cast<T (*)(T, U)>(&FUNCTION_NAME), boost::lambda::_1, index)); \
      //     return vec; \
      //   }

      //   IMPLEMENT_ALPS_VECTOR_FUNCTION2(std,pow)

      //   template <class T>
      //   std::ostream& operator<< (std::ostream &out, std::vector<T> const & vec)
      //   {
      //     std::copy(vec.begin(),vec.end(),std::ostream_iterator<T>(out,"\t"));
      //     return out;
      //   }

        //------------------- operator equal -------------------
        #define ALPS_NUMERIC_OPERATOR_EQ(OP_NAME, OPERATOR)                                                                 \
            template<typename T>                                                                                            \
            std::vector<T> & OP_NAME (std::vector<T> & lhs, std::vector<T> const & rhs) {                                   \
                if(lhs.size() != rhs.size())                                                                                \
                    boost::throw_exception(std::runtime_error("std::vectors must have the same size!" + ALPS_STACKTRACE));  \
                else                                                                                                        \
                    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std:: OPERATOR <T>() );                \
                return lhs;                                                                                                 \
            }

        ALPS_NUMERIC_OPERATOR_EQ(operator+=, plus)
        ALPS_NUMERIC_OPERATOR_EQ(operator-=, minus)
        ALPS_NUMERIC_OPERATOR_EQ(operator*=, multiplies)
        ALPS_NUMERIC_OPERATOR_EQ(operator/=, divides)

        #undef ALPS_NUMERIC_OPERATOR_EQ

        //------------------- infinity -------------------
        template<typename T> struct inf<std::vector<T> > {
            operator std::vector<T> const() {
                std::vector<T> retval;
                BOOST_FOREACH(T & arg, retval) {
                    arg = (T)inf<T>();
                }
                return retval;
            }
        };

        //------------------- unary operator - -------------------
        /// Unary negation of a vector
        template<typename T>
        std::vector<T> operator - (std::vector<T> lhs) {
            std::transform(lhs.begin(), lhs.end(), lhs.begin(), std::negate<T>());
            return lhs;
        }

        //------------------- operator + -------------------
        /// Sum of two vectors. Note: Treats a default-initialized vector (size 0) as 0-vector.
        template<typename T, typename U>
        std::vector<T> operator + (std::vector<T> const & lhs, std::vector<U> const & rhs) {
            using boost::numeric::operators::operator+;
            if (lhs.empty()) return rhs;
            if (rhs.empty()) return lhs;
            return lhs + rhs;
        }

        //------------------- operator - -------------------
        /// Difference of two vectors. Note: Treats a default-initialized vector (size 0) as 0-vector.
        template<typename T, typename U>
        std::vector<T> operator - (std::vector<T> const & lhs, std::vector<U> const & rhs) {
            using boost::numeric::operators::operator-;
            if (rhs.empty()) return lhs;
            if (lhs.empty()) return -rhs;
            return lhs - rhs;
        }

        //------------------- operator * vector-vector-------------------
        /// By-element product of two vectors. Note: Treats a default-initialized vector (size 0) as 0-vector.
        template<typename T, typename U>
        std::vector<T> operator * (std::vector<T> const & lhs, std::vector<U> const & rhs) {
            using boost::numeric::operators::operator*;
            if (lhs.empty()) return lhs;
            if (rhs.empty()) return rhs;
            return lhs * rhs;
        }
        //------------------- operator / vector-vector-------------------
        /// By-element quotient of two vectors. Note: Treats a default-initialized vector (size 0) as 0-vector.
        template<typename T, typename U>
        std::vector<T> operator / (std::vector<T> const & lhs, std::vector<U> const & rhs) {
            using boost::numeric::operators::operator/;
            if (lhs.empty()) return lhs;
            if (rhs.empty()) throw std::runtime_error("Division by default-initialized vector");
            return lhs / rhs;
        }

        //------------------- operator + with scalar -------------------
        /// Sum of a scalar and a vector
        template<typename T>
        std::vector<T> operator + (T const & scalar, std::vector<T> rhs) {
            std::transform(rhs.begin(), rhs.end(), rhs.begin(), bind1st(std::plus<T>(), scalar));
            return rhs;
        }
        /// Sum of a vector and a scalar
        template<typename T>
        std::vector<T> operator + (std::vector<T> lhs, T const & scalar) {
            std::transform(lhs.begin(), lhs.end(), lhs.begin(), bind2nd(std::plus<T>(), scalar));
            return lhs;
        }

        //------------------- operator - with scalar -------------------
        /// Difference of a scalar and a vector
        template<typename T>
        std::vector<T> operator - (T const & scalar, std::vector<T> const & rhs) {
            return scalar + -rhs;
        }
        /// Difference of a vector and a scalar
        template<typename T>
        std::vector<T> operator - (std::vector<T> const & lhs, T const & scalar) {
            return lhs + -scalar;
        }

        //------------------- operator * with scalar -------------------
        /// Returns a vector scaled by a scalar
        template<typename T>
        std::vector<T> operator * (std::vector<T> const & lhs, T const & scalar) {
            using boost::numeric::operators::operator*;
            return lhs * scalar;
        }
        /// Returns a vector scaled by a scalar
        template<typename T>
        std::vector<T> operator * (T const & scalar, std::vector<T> const & rhs) {
            using boost::numeric::operators::operator*;
            return scalar * rhs;
        }
        //------------------- operator / with scalar -------------------
        /// Returns a vector divided scaled by a scalar
        template<typename T>
        std::vector<T> operator / (std::vector<T> const & lhs, T const & scalar) {
            using boost::numeric::operators::operator/;
            return lhs / scalar;
        }
        /// Returns a vector with elements inverted and scaled by a scalar
        template<typename T>
        std::vector<T> operator / (T const & scalar, std::vector<T> rhs) {
            std::transform(rhs.begin(), rhs.end(), rhs.begin(), scalar / boost::lambda::_1);
            return rhs;
        }

        //------------------- numeric functions -------------------
        #define ALPS_NUMERIC_IMPLEMENT_FUNCTION(FUNCTION_NAME)                                                          \
            template<typename T> std::vector<T> FUNCTION_NAME (std::vector<T> arg) {                                    \
                using std:: FUNCTION_NAME ;                                                                             \
                std::transform(arg.begin(), arg.end(), arg.begin(), static_cast<T (*)(T)>(& FUNCTION_NAME )); \
                return arg;                                                                                             \
            }

        ALPS_NUMERIC_IMPLEMENT_FUNCTION(sin)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(cos)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(tan)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(sinh)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(cosh)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(tanh)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(asin)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(acos)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(atan)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(abs)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(sqrt)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(exp)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(log)

        #undef ALPS_NUMERIC_IMPLEMENT_FUNCTION

        template<typename T, typename U> std::vector<T> pow(std::vector<T> vec, U index) {
            using std::pow;
            std::transform(vec.begin(), vec.end(), vec.begin(), boost::lambda::bind<T>(static_cast<T (*)(T, T)>(&pow), boost::lambda::_1, index));
            return vec;
        }

        /// "Imported" negation functor class (needed to define template specializations in this namespace)
        template <typename T> struct negate: public std::negate<T> {};
        
        /// Negation for vectors (specialization of the standard template, here to avoid changes in std::)
        template <typename T>
        struct negate< std::vector<T> > {
            typedef std::vector<T> VT;
            VT operator()(VT v)
            {
                transform(v.begin(),v.end(),v.begin(),negate<T>());
                return v;
            }
        };
        
        /// A service functor class for numerical inversion, to be used in transform()
        template <typename T>
        struct invert {
            T operator()(T x) { return T(1.0)/x; }
        };

        /// Inversion for vectors, to be used in transform()
        template <typename T>
        struct invert< std::vector<T> > {
            typedef std::vector<T> VT;
            VT operator()(VT v)
            {
                transform(v.begin(),v.end(),v.begin(),invert<T>());
                return v;
            }
        };
        
    }
}

#endif // ALPS_NUMERIC_VECTOR_FUNCTIONS_HEADER




