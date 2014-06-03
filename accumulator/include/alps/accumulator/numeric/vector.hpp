/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef ALPS_NGS_NUMERIC_VECTOR_HEADER
#define ALPS_NGS_NUMERIC_VECTOR_HEADER

#include <alps/utility/stacktrace.hpp>
#include <alps/accumulator/numeric/inf.hpp>

#include <alps/numeric/vector_functions.hpp>
#include <alps/numeric/special_functions.hpp>

#include <boost/accumulators/numeric/functional/vector.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/throw_exception.hpp>
#include <boost/lambda/lambda.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/bind.hpp>

#include <vector>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace alps {
    namespace ngs {
        namespace numeric {

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
            template<> struct inf<std::vector<double> > {
                operator std::vector<double> const() {
                    std::vector<double> retval;
                    BOOST_FOREACH(double & arg, retval) {
                        arg = std::numeric_limits<double>::infinity();
                    }
                    return retval;
                }
            };

            //------------------- unary operator - -------------------
            template<typename T>
            std::vector<T> operator - (std::vector<T> lhs) {
                std::transform(lhs.begin(), lhs.end(), lhs.begin(), std::negate<T>());
                return lhs;
            }

            //------------------- operator + -------------------
            template<typename T, typename U>
            std::vector<T> operator + (std::vector<T> const & lhs, std::vector<U> const & rhs) {
                using boost::numeric::operators::operator+;
                return lhs + rhs;
            }
            //------------------- operator + with scalar -------------------
            template<typename T, std::size_t N>
            std::vector<T> operator + (std::vector<T> arg, T const & scalar) {
                std::transform(arg.begin(), arg.end(), arg.begin(), boost::lambda::_1 + scalar);
                return arg;
            }
            template<typename T, std::size_t N>
            std::vector<T> operator + (T const & scalar, std::vector<T> arg) {
                std::transform(arg.begin(), arg.end(), arg.begin(), scalar + boost::lambda::_1);
                return arg;
            }

            //------------------- operator - -------------------
            template<typename T, typename U>
            std::vector<T> operator - (std::vector<T> const & lhs, std::vector<U> const & rhs) {
                using boost::numeric::operators::operator-;
                return lhs - rhs;
            }
            //------------------- operator + with scalar -------------------
            template<typename T, std::size_t N>
            std::vector<T> operator - (std::vector<T> arg, T const & scalar) {
                std::transform(arg.begin(), arg.end(), arg.begin(), boost::lambda::_1 + scalar);
                return arg;
            }
            template<typename T, std::size_t N>
            std::vector<T> operator - (T const & scalar, std::vector<T> arg) {
                std::transform(arg.begin(), arg.end(), arg.begin(), scalar + boost::lambda::_1);
                return arg;
            }

            //------------------- operator * vector-vector-------------------
            template<typename T, typename U>
            std::vector<T> operator * (std::vector<T> const & lhs, std::vector<U> const & rhs) {
                using boost::numeric::operators::operator*;
                return lhs * rhs;
            }
            //------------------- operator / vector-vector-------------------
            template<typename T, typename U>
            std::vector<T> operator / (std::vector<T> const & lhs, std::vector<U> const & rhs) {
                using boost::numeric::operators::operator/;
                return lhs / rhs;
            }

            //------------------- operator + with scalar -------------------
            template<typename T>
            std::vector<T> operator + (T const & scalar, std::vector<T> lhs) {
                std::transform(lhs.begin(), lhs.end(), lhs.begin(), bind1st(std::plus<T>(), scalar));
                return lhs;
            }
            template<typename T>
            std::vector<T> operator + (std::vector<T> lhs, T const & scalar) {
                std::transform(lhs.begin(), lhs.end(), lhs.begin(), bind2nd(std::plus<T>(), scalar));
                return lhs;
            }

            //------------------- operator - with scalar -------------------
            template<typename T>
            std::vector<T> operator - (T const & scalar, std::vector<T> const & lhs) {
                return -scalar + lhs;
            }
            template<typename T>
            std::vector<T> operator - (std::vector<T> const & lhs, T const & scalar) {
                return lhs + -scalar;
            }

            //------------------- operator * with scalar -------------------
            template<typename T>
            std::vector<T> operator * (std::vector<T> const & lhs, T const & scalar) {
                using boost::numeric::operators::operator*;
                return lhs * scalar;
            }
            template<typename T>
            std::vector<T> operator * (T const & scalar, std::vector<T> const & rhs) {
                using boost::numeric::operators::operator*;
                return scalar * rhs;
            }
            //------------------- operator / with scalar -------------------
            template<typename T>
            std::vector<T> operator / (std::vector<T> const & lhs, T const & scalar) {
                using boost::numeric::operators::operator/;
                return lhs / scalar;
            }
            template<typename T>
            std::vector<T> operator / (T const & scalar, std::vector<T> rhs) {
                std::transform(rhs.begin(), rhs.end(), rhs.begin(), scalar / boost::lambda::_1);
                return rhs;
            }

            //------------------- numeric functions -------------------
            #define ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(FUNCTION_NAME)                                                      \
                template<typename T> std::vector<T> FUNCTION_NAME (std::vector<T> arg) {                                    \
                    using std:: FUNCTION_NAME ;                                                                             \
                    std::transform(arg.begin(), arg.end(), arg.begin(), static_cast<double (*)(double)>(& FUNCTION_NAME )); \
                    return arg;                                                                                             \
                }

            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(sin)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(cos)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(tan)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(sinh)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(cosh)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(tanh)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(asin)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(acos)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(atan)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(abs)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(sqrt)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(exp)
            ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(log)

            #undef ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION

            // #define ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(FUNCTION_NAME)                                                      \
            //     template<typename T> std::vector<T> FUNCTION_NAME (std::vector<T> arg) {                                    \
            //         using alps::numeric:: FUNCTION_NAME ;                                                                   \
            //         std::transform(arg.begin(), arg.end(), arg.begin(), static_cast<double (*)(double)>(& FUNCTION_NAME )); \
            //         return arg;                                                                                             \
            //     }

            // ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(sq)
            // ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(cb)
            // ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION(cbrt)

            #undef ALPS_NGS_NUMERIC_IMPLEMENT_FUNCTION

            template<typename T, typename U> std::vector<T> pow(std::vector<T> vec, U index) {
                using std::pow;
                std::transform(vec.begin(), vec.end(), vec.begin(), boost::lambda::bind<T>(static_cast<T (*)(T, U)>(&pow), boost::lambda::_1, index));
                return vec;
            }
        }
    }
}


#endif
