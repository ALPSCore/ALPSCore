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
#ifndef ALPS_NUMERIC_OPERATORS_OP_ASSIGN_HPP
#define ALPS_NUMERIC_OPERATORS_OP_ASSIGN_HPP

#include <boost/static_assert.hpp>

namespace alps {
namespace numeric {

template <typename T>
struct not_implemented
{
    static bool const value = false;
};

template <typename T1, typename T2, typename Category1, typename Category2>
void plus_assign(T1& t1, T2 const& t2, Category1, Category2)
{
    BOOST_STATIC_ASSERT(not_implemented<T1>::value);
}

template <typename T1, typename T2, typename Category1, typename Category2>
void minus_assign(T1& t1, T2 const& t2, Category1, Category2)
{
    BOOST_STATIC_ASSERT(not_implemented<T1>::value);
}

template <typename T1, typename T2, typename Category1, typename Category2>
void multiplies_assign(T1& t1, T2 const& t2, Category1, Category2)
{
    BOOST_STATIC_ASSERT(not_implemented<T1>::value);
}

} // end namespace numeric
} // end namespace alps

#endif // ALPS_NUMERIC_OPERATORS_OP_ASSIGN_HPP
