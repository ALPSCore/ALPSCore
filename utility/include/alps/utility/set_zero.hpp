/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_UTILITY_SET_ZERO_HPP
#define ALPS_UTILITY_SET_ZERO_HPP

#include <alps/type_traits/is_sequence.hpp>
#include <alps/type_traits/element_type.hpp>

#include <boost/utility/enable_if.hpp>

#include <algorithm>

namespace alps {

template <class X> 
inline typename boost::disable_if<is_sequence<X>,void>::type
set_zero(X& x) { x=X();}

template <class X> 
inline typename boost::enable_if<is_sequence<X>,void>::type
set_zero(X& x) 
{
  std::fill(x.begin(),x.end(),typename element_type<X>::type());
}



} // end namespace alps

#endif // ALPS_UTILITY_RESIZE_HPP
