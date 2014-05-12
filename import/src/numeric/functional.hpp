/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Lukas Gamper <gamperl@gmail.com>
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

/* $Id: funcitonal.hpp 3958 2010-03-05 09:24:06Z gamperl $ */

#ifndef ALPS_NUMERIC_FUNCTIONAL_HPP
#define ALPS_NUMERIC_FUNCTIONAL_HPP

#include <alps/numeric/vector_functions.hpp>
#include <alps/boost/accumulators/numeric/functional.hpp>
#include <alps/boost/accumulators/numeric/functional/vector.hpp>

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
                using boost::numeric::operators::operator+;
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
                using boost::numeric::operators::operator-;
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
                using boost::numeric::operators::operator*;
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
                using boost::numeric::operators::operator/;
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
