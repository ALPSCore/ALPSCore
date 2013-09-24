/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2013 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
*                            Synge Todo <wistaria@comp-phys.org>,
*                            Andreas Hehn <hehn@itp.phys.ethz.ch>
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

#ifndef ALPS_NUMERIC_VALARRAY_SCALAR_PRODUCT_HPP
#define ALPS_NUMERIC_VALARRAY_SCALAR_PRODUCT_HPP

#include <alps/functional.h>
#include <functional>
#include <numeric>

// Some fwd declaration
namespace std {
    template <class T>
    class valarray;
}

namespace alps { namespace numeric {

/// \overload for valarrays
#if defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0) || defined(BOOST_MSVC)
// Workaround for a compiler bug in clang 3.0 (and maybe earlier versions)
template <class T>
inline T scalar_product(const std::valarray<T>& c1, const std::valarray<T>& c2) 
{
  return std::inner_product(data(c1),data(c1)+c1.size(),data(c2),T(), std::plus<T>(),conj_mult<T,T>());
}
#else // defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)
template <class T>
inline T scalar_product(const std::valarray<T>& c1, const std::valarray<T>& c2) 
{
  return std::inner_product(data(c1),data(c1)+c1.size(),data(c2),T(), std::plus<T>(),conj_mult<T,T>());
}
#endif // defined(__clang_major__) && __clang_major__ < 3 || (__clang_major__ == 3 && __clang_minor__ == 0)

} } // namespace alps::numeric

#endif // ALPS_NUMERIC_VALARRAY_SCALAR_PRODUCT_HPP
