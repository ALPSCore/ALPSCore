/*
 * Copyright (C) 1998-2017 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_UTILITY_SIZE_HPP
#define ALPS_UTILITY_SIZE_HPP

#include <alps/type_traits/is_sequence.hpp>
#include <boost/utility/enable_if.hpp>

namespace alps {

template <class T>
inline typename boost::disable_if<is_sequence<T>,std::size_t>::type
size(T const&) 
{
  return 1;
}

template <class T>
inline typename boost::enable_if<is_sequence<T>,std::size_t>::type
size(T const& a) 
{
  return a.size();
}

} // end namespace alps

#endif // ALPS_UTILITY_SIZE_HPP
