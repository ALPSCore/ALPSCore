/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2010 by Matthias Troyer <troyer@comp-phys.org>,
*                            Beat Ammon <ammon@ginnan.issp.u-tokyo.ac.jp>,
*                            Andreas Laeuchli <laeuchli@comp-phys.org>,
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

#ifndef ALPS_UTILITY_RESIZE_HPP
#define ALPS_UTILITY_RESIZE_HPP

#include <alps/type_traits/is_sequence.hpp>
#include <alps/multi_array.hpp>

#include <boost/mpl/or.hpp>
#include <boost/mpl/and.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/array.hpp>

#include <algorithm>

namespace alps {

template <class X, class Y> 
inline typename boost::disable_if<boost::mpl::or_<is_sequence<X>,is_sequence<Y> >,void>::type
resize_same_as(X&, const Y&) {}

template <class X, class Y> 
inline typename boost::enable_if<boost::mpl::and_<is_sequence<X>,is_sequence<Y> >,void>::type
resize_same_as(X& a, const Y& y) 
{
  a.resize(y.size());
}

template<typename T, typename U, std::size_t N>
inline void resize_same_as(alps::multi_array<T, N> & a, alps::multi_array<U, N> const & y)
{
    const typename alps::multi_array<T, N>::size_type* shp = y.shape();
    std::vector<typename alps::multi_array<T, N>::size_type> ext(shp,shp + y.num_dimensions());
    a.resize(ext);
}

template<typename T, typename U, std::size_t N>
inline void resize_same_as(boost::array<T, N> & a, boost::array<U, N> const & y)
{
}

} // end namespace alps

#endif // ALPS_UTILITY_RESIZE_HPP
