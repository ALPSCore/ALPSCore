/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_CHECKED_DIVIDE_HPP
#define ALPS_NUMERIC_CHECKED_DIVIDE_HPP

#include <alps/type_traits/is_sequence.hpp>

#include <type_traits>

namespace alps { namespace numeric {



template <class T>
inline typename std::enable_if<!is_sequence<T>::value,T>::type
checked_divide(const T& a,const T& b)
{
  return (b==T() && a==T()? 1. : a/b);
}

template <class T>
inline typename std::enable_if<is_sequence<T>::value,T>::type
checked_divide(T a,const T& b)
{
  for(std::size_t i=0;i<b.size();++i)
    a[i] = checked_divide(a[i],b[i]);
  return a;
}


} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_CHECKED_DIVIDE_HPP
