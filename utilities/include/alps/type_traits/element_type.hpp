/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_ELEMENT_TYPE_H
#define ALPS_TYPE_TRAITS_ELEMENT_TYPE_H

#include <alps/type_traits/has_value_type.hpp>

#include <type_traits>

 namespace alps {

template <class T> struct element_type_recursive;

namespace detail {

template <class T, class F>
struct element_type_helper {};

 template <class T>
struct element_type_helper<T,std::false_type>
{
  typedef T type;
};

template <class T>
struct element_type_helper<T,std::true_type>
{
    typedef typename T::value_type type;
};

template <class T, class F>
struct element_type_recursive_helper {};

 template <class T>
struct element_type_recursive_helper<T,std::false_type>
{
  typedef T type;
};

template <class T>
struct element_type_recursive_helper<T,std::true_type>
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
