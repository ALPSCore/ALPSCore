/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2002-2003 by Synge Todo <wistaria@comp-phys.org>
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

#ifndef ALPS_FIXED_CAPACITY_FWD_H
#define ALPS_FIXED_CAPACITY_FWD_H

#include <cstddef>

namespace alps {

namespace fixed_capacity {

// Forward declarations of checking policy classes
//   definitions are given in <alps/fixed_capacity/checking.h>

struct no_checking;
struct capacity_checking;
struct strict_checking;

} // namespace fixed_capacity

// Forward declarations of alps::fixed_capacity_[vector,deque]
//   definitions are given in <alps/fixed_capacity_vector.h> and
//   <alps/fixed_capacity_deque.h>, respectively

template<class T,
         std::size_t N,
         class CheckingPolicy = ::alps::fixed_capacity::no_checking>
class fixed_capacity_vector;
template<class T,
         std::size_t N,
         class CheckingPolicy = ::alps::fixed_capacity::no_checking>
class fixed_capacity_deque;

// Forward declaration of traits class alps::fixed_capacity_traits
//   definition is given in <alps/fixed_capacity_traits.h>

template<class C> struct fixed_capacity_traits;

} // namespace alps

#endif // ALPS_FIXED_CAPACITY_FWD_H
