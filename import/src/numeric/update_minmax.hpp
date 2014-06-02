/*
 * Copyright (C) 1998-2014 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id$ */

#ifndef ALPS_NUMERIC_UPDATE_MINMAX_HPP
#define ALPS_NUMERIC_UPDATE_MINMAX_HPP

#include <alps/type_traits/slice.hpp>

namespace alps { namespace numeric {

template <class T>
void update_max(T& lhs, T const& rhs)
{
  for (typename slice_index<T>::type it = slices(lhs).first; 
       it < slices(lhs).second && it < slices(rhs).second; ++it)
    if (slice_value(lhs,it) < slice_value(rhs,it))
      slice_value(lhs,it) = slice_value(rhs,it);
}

template <class T>
void update_min(T& lhs, T const& rhs)
{
  for (typename slice_index<T>::type it = slices(lhs).first; 
       it < slices(lhs).second && it < slices(rhs).second; ++it)
    if (slice_value(rhs,it) < slice_value(lhs,it))
      slice_value(lhs,it) = slice_value(rhs,it);
}



} } // end namespace alps::numeric

#endif // ALPS_NUMERIC_UPDATE_MINMAX_HPP
