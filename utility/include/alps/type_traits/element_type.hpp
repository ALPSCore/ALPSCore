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

#ifndef ALPS_TYPE_TRAITS_ELEMENT_TYPE_H
#define ALPS_TYPE_TRAITS_ELEMENT_TYPE_H

#include <alps/type_traits/has_value_type.hpp>
#include <boost/mpl/eval_if.hpp>
#include <boost/mpl/bool.hpp>
 
 namespace alps {
 
template <class T> struct element_type_recursive;

namespace detail {

template <class T, class F>
struct element_type_helper {};
  
 template <class T>
struct element_type_helper<T,boost::mpl::false_> 
{
  typedef T type;
};

template <class T>
struct element_type_helper<T,boost::mpl::true_> 
{
    typedef typename T::value_type type;
};

template <class T, class F>
struct element_type_recursive_helper {};
  
 template <class T>
struct element_type_recursive_helper<T,boost::mpl::false_> 
{
  typedef T type;
};

template <class T>
struct element_type_recursive_helper<T,boost::mpl::true_> 
    : element_type_recursive<typename T::value_type>
{
};


}

template <class T>
 struct element_type
 : public detail::element_type_helper<T,typename has_value_type<T>::type > {};

 template <class T>
 struct element_type_recursive
 : public detail::element_type_recursive_helper<T,typename has_value_type<T>::type > {};

}
 
#endif // ALPS_TYPE_TRAITS_ELEMENT_TYPE_H
