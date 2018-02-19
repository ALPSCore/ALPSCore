/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_CHANGE_VALUE_TYPE_H
#define ALPS_TYPE_TRAITS_CHANGE_VALUE_TYPE_H

#include <alps/config.hpp>
#include <valarray>
#include <vector>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T, class V>
struct change_value_type
{
  typedef V type;
};

template <class T, class A, class V>
struct change_value_type<std::vector<T,A>,V>
{
  typedef std::vector<V> type;
};

template <class T, class V>
struct change_value_type<std::valarray<T>,V>
{
  typedef std::valarray<V> type;
};

template <class T, class V>
struct change_value_type_replace_valarray : change_value_type<T,V> {};

template <class T, class V>
struct change_value_type_replace_valarray<std::valarray<T>,V>
{
  typedef std::vector<V> type;
};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_CHANGE_VALUE_TYPE_H
