/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2010 by Matthias Troyer <troyer@comp-phys.org>,
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

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_PARAM_TYPE_H
#define ALPS_TYPE_TRAITS_PARAM_TYPE_H

#include <alps/type_traits/is_scalar.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <typename T> 
struct param_type : public boost::mpl::if_<
      typename is_scalar<typename boost::remove_cv<T>::type>::type
    , T
    , typename boost::add_reference<typename boost::add_const<T>::type>::type
> {};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_NORM_TYPE_H
