/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NUMERIC_ARRAY_HEADER
#define ALPS_NUMERIC_ARRAY_HEADER

#include <alps/utilities/stacktrace.hpp>

#include <alps/numeric/inf.hpp>
#include <alps/numeric/special_functions.hpp>

#include <boost/throw_exception.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/array.hpp>

#include <algorithm>
#include <functional>
#include <cmath>
#include <stdexcept>

namespace alps {
    namespace numeric {

        //------------------- operator equal -------------------
        #define ALPS_NUMERIC_OPERATOR_EQ(OP_NAME, OPERATOR)                                                             \
            template<typename T, std::size_t N>                                                                         \
            boost::array<T, N> & OP_NAME (boost::array<T, N> & lhs, boost::array<T, N> const & rhs) {                   \
                if(lhs.size() != rhs.size())                                                                            \
                    boost::throw_exception(std::runtime_error("arrays must have the same size!" + ALPS_STACKTRACE));    \
                else                                                                                                    \
                    std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std:: OPERATOR <T>() );            \
                return lhs;                                                                                             \
            }

        ALPS_NUMERIC_OPERATOR_EQ(operator+=, plus)
        ALPS_NUMERIC_OPERATOR_EQ(operator-=, minus)
        ALPS_NUMERIC_OPERATOR_EQ(operator*=, multiplies)
        ALPS_NUMERIC_OPERATOR_EQ(operator/=, divides)

        #undef ALPS_NUMERIC_OPERATOR_EQ

        //------------------- infinity -------------------
        template<std::size_t N> struct inf<boost::array<double, N> > {
            operator boost::array<double, N> const() {
                boost::array<double, N> retval;
                for(double & arg: retval) {
                    arg = std::numeric_limits<double>::infinity();
                }
                return retval;
            }
        };

        //------------------- unary operator - -------------------
        template<typename T, std::size_t N>
        boost::array<T, N> operator - (boost::array<T, N> lhs) {
            std::transform(lhs.begin(), lhs.end(), lhs.begin(), std::negate<T>());
            return lhs;
        }

        //------------------- operator + -------------------
        template<typename T, typename U, std::size_t N>
        boost::array<T, N> operator + (boost::array<T, N> lhs, boost::array<U, N> const & rhs) {
            std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::plus<T>() );
            return lhs;
        }
        //------------------- operator + with scalar -------------------
        template<typename T, std::size_t N>
        boost::array<T, N> operator + (boost::array<T, N> arg, T const & scalar) {
            std::transform(arg.begin(), arg.end(), arg.begin(), boost::lambda::_1 + scalar);
            return arg;
        }
        template<typename T, std::size_t N>
        boost::array<T, N> operator + (T const & scalar, boost::array<T, N> arg) {
            std::transform(arg.begin(), arg.end(), arg.begin(), scalar + boost::lambda::_1);
            return arg;
        }

        //------------------- operator - -------------------
        template<typename T, typename U, std::size_t N>
        boost::array<T, N> operator - (boost::array<T, N> lhs, boost::array<U, N> const & rhs) {
            std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::minus<T>() );
            return lhs;
        }
        //------------------- operator - with scalar -------------------
        template<typename T, std::size_t N>
        boost::array<T, N> operator - (boost::array<T, N> arg, T const & scalar) {
            std::transform(arg.begin(), arg.end(), arg.begin(), boost::lambda::_1 - scalar);
            return arg;
        }
        template<typename T, std::size_t N>
        boost::array<T, N> operator - (T const & scalar, boost::array<T, N> arg) {
            std::transform(arg.begin(), arg.end(), arg.begin(), scalar - boost::lambda::_1);
            return arg;
        }

        //------------------- operator * vector-vector-------------------
        template<typename T, typename U, std::size_t N>
        boost::array<T, N> operator * (boost::array<T, N> lhs, boost::array<U, N> const & rhs)
        {
            std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::multiplies<T>());
            return lhs;
        }
        //------------------- operator / vector-vector-------------------
        template<typename T, typename U, std::size_t N>
        boost::array<T, N> operator / (boost::array<T, N> lhs, boost::array<U, N> const & rhs)
        {
            std::transform(lhs.begin(), lhs.end(), rhs.begin(), lhs.begin(), std::multiplies<T>());
            return lhs;
        }

        //------------------- operator * with scalar -------------------
        template<typename T, std::size_t N>
        boost::array<T, N> operator * (boost::array<T, N> lhs, T const & scalar) {
            std::transform(lhs.begin(), lhs.end(), lhs.begin(), boost::lambda::_1 * scalar);
            return lhs;
        }
        template<typename T, std::size_t N>
        boost::array<T, N> operator * (T const & scalar, boost::array<T, N> rhs) {
            std::transform(rhs.begin(), rhs.end(), rhs.begin(), scalar * boost::lambda::_1);
            return rhs;
        }

        //------------------- operator / with scalar -------------------
        template<typename T, std::size_t N>
        boost::array<T, N> operator / (boost::array<T, N> lhs, T const & scalar) {
            std::transform(lhs.begin(), lhs.end(), lhs.begin(), boost::lambda::_1 / scalar);
            return lhs;
        }
        template<typename T, std::size_t N>
        boost::array<T, N> operator / (T const & scalar, boost::array<T, N> rhs) {
            std::transform(rhs.begin(), rhs.end(), rhs.begin(), scalar / boost::lambda::_1);
            return rhs;
        }

        //------------------- numeric functions -------------------
        #define ALPS_NUMERIC_IMPLEMENT_FUNCTION(FUNCTION_NAME)                                                      \
            template<typename T, std::size_t N> boost::array<T, N> FUNCTION_NAME (boost::array<T, N> arg) {             \
                using std:: FUNCTION_NAME ;                                                                             \
                std::transform(arg.begin(), arg.end(), arg.begin(), static_cast<double (*)(double)>(& FUNCTION_NAME )); \
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

        #define ALPS_NUMERIC_IMPLEMENT_FUNCTION(FUNCTION_NAME)                                                      \
            template<typename T, std::size_t N> boost::array<T, N> FUNCTION_NAME (boost::array<T, N> arg) {             \
                using alps::numeric:: FUNCTION_NAME ;                                                                   \
                std::transform(arg.begin(), arg.end(), arg.begin(), static_cast<double (*)(double)>(& FUNCTION_NAME )); \
                return arg;                                                                                             \
            }

        ALPS_NUMERIC_IMPLEMENT_FUNCTION(sq)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(cb)
        ALPS_NUMERIC_IMPLEMENT_FUNCTION(cbrt)

        #undef ALPS_NUMERIC_IMPLEMENT_FUNCTION

    }
}

#endif
