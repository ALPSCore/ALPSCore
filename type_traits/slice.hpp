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

#ifndef ALPS_TYPE_TRAITS_SLICE_H
#define ALPS_TYPE_TRAITS_SLICE_H

#include <alps/type_traits/element_type.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/lexical_cast.hpp>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct slice_index
{
  typedef std::size_t type;
};

template <class T>
typename boost::enable_if<is_sequence<T>,
  std::pair<typename slice_index<T>::type,typename slice_index<T>::type >
>::type
slices (T const& x) 
{ 
  return std::pair<typename slice_index<T>::type,typename slice_index<T>::type >(0,x.size());
}

template <class T>
typename boost::disable_if<is_sequence<T>,
  std::pair<typename slice_index<T>::type,typename slice_index<T>::type >
>::type
slices (T const&) 
{ 
  return std::pair<typename slice_index<T>::type,typename slice_index<T>::type >(0,1);
}

template <class ValueType, class SliceIndex>
typename boost::enable_if<is_sequence<ValueType>, std::string>::type
slice_name(ValueType const& ,SliceIndex i) 
{ 
  return boost::lexical_cast<std::string,int>(i);
}

template <class ValueType, class SliceIndex>
typename boost::disable_if<is_sequence<ValueType>, std::string>::type
slice_name(ValueType const& ,SliceIndex) 
{ 
  return "";
}

template <class ValueType, class SliceIndex>
typename boost::enable_if<is_sequence<ValueType>, 
  typename element_type<ValueType>::type>::type
slice_value(ValueType const& x ,SliceIndex i) 
{ 
  return i < x.size() ? x[i] : typename element_type<ValueType>::type();
}


template <class ValueType, class SliceIndex>
typename boost::disable_if<is_sequence<ValueType>, ValueType const&>::type
slice_value(ValueType const& x ,SliceIndex) 
{ 
  return x;
}

template <class ValueType, class SliceIndex>
typename boost::enable_if<is_sequence<ValueType>, 
  typename element_type<ValueType>::type&>::type
slice_value(ValueType& x,SliceIndex i) 
{ 
  return x[i];
}


template <class ValueType, class SliceIndex>
typename boost::disable_if<is_sequence<ValueType>, ValueType&>::type
slice_value(ValueType& x ,SliceIndex) 
{ 
  return x;
}

template <class ValueType>
struct slice_it  
{                      
  typedef typename element_type<ValueType>::type result_type;
  typedef ValueType const& first_argument_type;
  typedef typename slice_index<ValueType>::type second_argument_type;
   
  result_type operator()(ValueType const& x, second_argument_type i) const
  { 
    return slice_value(x,i);
  }
};                                             


} // end namespace alps

#endif // ALPS_TYPE_TRAITS_ELEMENT_TYPE_H
