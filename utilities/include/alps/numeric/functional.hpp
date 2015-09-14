/*
 * Copyright (C) 1998-2015 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: funcitonal.hpp 3958 2010-03-05 09:24:06Z gamperl $ */

#ifndef ALPS_NUMERIC_FUNCTIONAL_HPP
#define ALPS_NUMERIC_FUNCTIONAL_HPP

#include <alps/numeric/vector_functions.hpp>
#include <boost/accumulators/numeric/functional.hpp>
#include <boost/accumulators/numeric/functional/vector.hpp>

namespace alps { 
    namespace numeric {
        template <typename T> struct unary_minus : public std::unary_function<T, T> {
            T operator()(T const & x) const {
                using boost::numeric::operators::operator-;
                return -x; 
            }
        };

        template <typename T, typename U, typename R> struct plus : public std::binary_function<T, U, R> {
            R operator()(T const & x, U const & y) const {
                // using boost::numeric::operators::operator+;
                using alps::numeric::operator+;
                return x + y; 
            }
        };
        template <typename T> struct plus<T, T, T> : public std::binary_function<T, T, T> {
            T operator()(T const & x, T const & y) const {
                using boost::numeric::operators::operator+;
                return x + y; 
            }
        };

        template <typename T, typename U, typename R> struct minus : public std::binary_function<T, U, R> {
            R operator()(T const & x, U const & y) const {
                // using boost::numeric::operators::operator-;
                using alps::numeric::operator-;
                return x - y; 
            }
        };
        template <typename T> struct minus<T, T, T> : public std::binary_function<T, T, T> {
            T operator()(T const & x, T const & y) const {
                using boost::numeric::operators::operator-;
                return x - y; 
            }
        };

        template <typename T, typename U, typename R> struct multiplies : public std::binary_function<T, U, R> {
            R operator()(T const & x, U const & y) const {
                // using boost::numeric::operators::operator*;
                using alps::numeric::operator*;
                return x * y; 
            }
        };
        template <typename T> struct multiplies<T, T, T> : public std::binary_function<T, T, T> {
            T operator()(T const & x, T const & y) const {
                using boost::numeric::operators::operator*;
                return x * y; 
            }
        };

        template <typename T, typename U, typename R> struct divides : public std::binary_function<T, U, R> {
            R operator()(T const & x, U const & y) const {
                // using boost::numeric::operators::operator/;
                using alps::numeric::operator/;
                return x / y; 
            }
        };
        template <typename T> struct divides<T, T, T> : public std::binary_function<T, T, T> {
            T operator()(T const & x, T const & y) const {
                using boost::numeric::operators::operator/;
                return x / y; 
            }
        };
    } 
}

#endif // ALPS_NUMERIC_FUNCTIONAL_HPP
