/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 1999-2010 by Matthias Troyer <troyer@itp.phys.ethz.ch>,
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

#ifndef ALPS_TYPE_TRAITS_ITERATOR_TYPE_HPP
#define ALPS_TYPE_TRAITS_ITERATOR_TYPE_HPP

#include <valarray>

namespace alps {

template <class Collection> 
struct iterator_type
{
  typedef typename Collection::iterator type;
};

template <class Collection> 
struct const_iterator_type
{
  typedef typename Collection::iterator type;
};

template <class T> 
struct iterator_type<std::valarray<T> >
{
  typedef T* type;
};

template <class T> 
struct const_iterator_type<std::valarray<T> >
{
  typedef T const * type;
};

} // namespace alps

#endif // ALPS_TYPE_TRAITS_ITERATOR_TYPE_HPP
