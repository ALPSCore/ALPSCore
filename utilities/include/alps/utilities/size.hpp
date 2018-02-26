/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_UTILITY_SIZE_HPP
#define ALPS_UTILITY_SIZE_HPP

#include <alps/type_traits/is_sequence.hpp>

#include <type_traits>

namespace alps {

template <class T>
inline typename std::enable_if<!is_sequence<T>::value,std::size_t>::type
size(T const&)
{
  return 1;
}

template <class T>
inline typename std::enable_if<is_sequence<T>::value,std::size_t>::type
size(T const& a)
{
  return a.size();
}

} // end namespace alps

#endif // ALPS_UTILITY_SIZE_HPP
