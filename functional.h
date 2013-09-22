/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1994-2005 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

/// \file functional.h
/// \brief extensions to the standard functional header
///
/// This header contains mathematical function objects not present in
/// the standard or boost libraries.

#ifndef ALPS_FUNCTIONAL_H
#define ALPS_FUNCTIONAL_H

#include <alps/numeric/matrix/detail/auto_deduce_multiply_return_type.hpp>
#include <alps/numeric/conj.hpp>

namespace alps {


template <class T1, class T2>
struct conj_mult_return_type : ::alps::numeric::detail::auto_deduce_multiply_return_type<T1,T2> { };

/// \brief a function object for conj(x)*y
///
/// the version for real data types is just the same as std::multiplies
/// \param T the type of the arguments and result
template <class T1, class T2>
struct conj_mult
{
/// \brief returns x*y
  typename conj_mult_return_type<T1,T2>::type operator()(const T1& a, const T2& b) const { return a*b; }
};

/// \brief a function object for conj(x)*y
///
/// the version for complex data types is specialized
/// \param T the type of the arguments and result
template <class T1, class T2>
struct conj_mult<std::complex<T1>,T2>
{
/// \brief returns std::conj(x)*y
  typename conj_mult_return_type<std::complex<T1>,T2>::type operator()(const std::complex<T1>& a, const T2& b) const {
  return std::conj(a)*b;
}
};

}

#endif // ALPS_FUNCTIONAL_H
