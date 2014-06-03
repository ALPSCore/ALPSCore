/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

#ifndef PARAPACK_FOOTPRINT_H
#define PARAPACK_FOOTPRINT_H

#include <alps/config.h>
#include <boost/type_traits.hpp>
#include <boost/utility/enable_if.hpp>

#include <string>
#include <vector>

namespace alps {

template<typename T>
std::size_t footprint(T const& t, typename boost::enable_if<boost::is_pod<T> >::type* = 0) {
  return sizeof(T);
}

template<typename T>
std::size_t footprint(T const& t, typename boost::disable_if<boost::is_pod<T> >::type* = 0) {
  return t.footprint();
}

template<typename T>
std::size_t footprint(std::vector<T> const& v) {
  return sizeof(std::vector<T>) + sizeof(T) * v.capacity();
}

template<typename C, typename T, typename A>
std::size_t footprint(std::basic_string<C, T, A> const& v) {
  return sizeof(std::basic_string<C, T, A>) + sizeof(C) * v.capacity();
}

} // end namespace alps

#endif // PARAPACK_FOOTPRINT_H
