/*
 * Copyright (C) 1998-2018 ALPS Collaboration. See COPYRIGHT.TXT
 * All rights reserved. Use is subject to license terms. See LICENSE.TXT
 * For use in publications, see ACKNOWLEDGE.TXT
 */

/* $Id: obsvalue.h 3435 2009-11-28 14:45:38Z troyer $ */

#ifndef ALPS_TYPE_TRAITS_AVERGAE_TYPE_H
#define ALPS_TYPE_TRAITS_AVERGAE_TYPE_H

#include <type_traits>
#include <valarray>
#include <vector>

// maybe we can automate this by checking for the existence of a value_type member

namespace alps {

template <class T>
struct average_type
 : public std::conditional<std::is_integral<T>::value,double,T> {};

template <class T>
struct average_type<std::valarray<T> > {
  typedef std::valarray<typename average_type<T>::type> type;
};

template <class T, class A>
struct average_type<std::vector<T,A> > {
  typedef std::vector<typename average_type<T>::type,A> type;
};

} // end namespace alps

#endif // ALPS_TYPE_TRAITS_AVERGAE_TYPE_H
