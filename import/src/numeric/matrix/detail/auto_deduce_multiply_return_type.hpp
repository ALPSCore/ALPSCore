/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *                                                                                 *
 * ALPS Project: Algorithms and Libraries for Physics Simulations                  *
 *                                                                                 *
 * ALPS Libraries                                                                  *
 *                                                                                 *
 * Copyright (C) 2013 by Andreas Hehn <hehn@phys.ethz.ch>                          *
 *                                                                                 *
 * This software is part of the ALPS libraries, published under the ALPS           *
 * Library License; you can use, redistribute it and/or modify it under            *
 * the terms of the license, either version 1 or (at your option) any later        *
 * version.                                                                        *
 *                                                                                 *
 * You should have received a copy of the ALPS Library License along with          *
 * the ALPS Libraries; see the file LICENSE.txt. If not, the license is also       *
 * available from http://alps.comp-phys.org/.                                      *
 *                                                                                 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR     *
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        *
 * FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT       *
 * SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE       *
 * FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE,     *
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER     *
 * DEALINGS IN THE SOFTWARE.                                                       *
 *                                                                                 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef ALPS_NUMERIC_MATRIX_DETAIL_AUTO_DEDUCE_MULTIPLY_RETURN_TYPE_HPP
#define ALPS_NUMERIC_MATRIX_DETAIL_AUTO_DEDUCE_MULTIPLY_RETURN_TYPE_HPP

#include <boost/mpl/if.hpp>

namespace alps {
namespace numeric {
namespace detail {

template <typename T1, typename T2>
struct auto_deduce_multiply_return_type
{
    private:
        typedef char one;
        typedef long unsigned int two;
        static one test(T1 t) {return one();}
        static two test(T2 t) {return two();}
    public:
        typedef boost::mpl::bool_<(sizeof(test(T1()*T2())) == sizeof(one))> select_first;
        typedef typename boost::mpl::if_<select_first,T1,T2>::type type;
};

template <typename T>
struct auto_deduce_multiply_return_type<T,T>
{
    typedef boost::mpl::bool_<true> select_first;
    typedef T type;
};

} // end namespace detail
} // end namespace numeric
} // end namespace alps
#endif // ALPS_NUMERIC_MATRIX_DETAIL_AUTO_DEDUCE_MULTIPLY_RETURN_TYPE_HPP
